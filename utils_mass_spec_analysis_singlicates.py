"""
utils_mass_spec_analysis_singlicates.py

A class-based refactoring of the mass spec data loading and preprocessing,
including widget-based sample renaming and data rebuilding. Designed for use
in Jupyter notebooks.
"""

import sys
import os
import json
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go


class MassSpecPreprocessing:
    def __init__(
        self,
        mass_spec_file_path,
        poi_file_path,
        color_code_file_path,
        abundance_column_label,
        genes_col_letter,
        accession_col_letter,
        description_col_letter,
        required_path=r"\\storage.imp.ac.at\groups\plaschka\shared\data\mass-spec"

    ):
        """
        Initialize all paths and parameters needed for preprocessing.
        """
        # Parameters from the user/notebook
        self.mass_spec_file_path = mass_spec_file_path
        self.poi_file_path = poi_file_path
        self.color_code_file_path = color_code_file_path
        self.abundance_column_label = abundance_column_label
        self.genes_col_letter = genes_col_letter
        self.accession_col_letter = accession_col_letter
        self.description_col_letter = description_col_letter
        self.required_path = required_path

        # Internal attributes that will be set during processing
        self.mass_spec_data_full_headers = None
        self.mass_spec_data = None
        self.genes_column = None
        self.accession_column = None
        self.description_column = None
        self.area_norm_columns = None
        self.sample_names_extracted = None
        self.mass_spec_abundance = None
        self.merged_data = None
        self.color_map = None
        self.abundance_cols = None  # renamed final sample columns

        # Widgets and outputs
        self.apply_button = widgets.Button(
            description="Apply New Names & Rebuild Data", button_style="success"
        )
        self.apply_output = widgets.Output()

        # Where to store the rename map JSON
        self.json_file_path = os.path.join("JSON", "sample_rename_map.json")

    def install_packages(self, packages):
        """
        Install required packages if they are not already present.
        """
        for package in packages:
            try:
                __import__(package)
            except ImportError:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def column_letter_to_index(self, column_letter):
        """
        Convert Excel-style column letters (e.g. 'A', 'B', 'AA') to zero-based indices.
        """
        column_letter = column_letter.upper()
        index = 0
        for char in column_letter:
            index = index * 26 + (ord(char) - ord('A') + 1)
        return index - 1

    def get_column_by_letter(self, data, column_letter):
        """
        Extract a column from a DataFrame by an Excel-style letter.
        Returns (column_series, column_name).
        """
        col_idx = self.column_letter_to_index(column_letter)
        col_name = data.columns[col_idx]
        return data[col_name], col_name

    def load_rename_map(self, json_path):
        """
        Load a JSON rename map (old sample name -> new sample name). Returns None if missing.
        """
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r') as f:
            return json.load(f)

    def save_rename_map(self, json_path, rename_dict):
        """
        Save the rename map dictionary to a JSON file.
        """
        with open(json_path, 'w') as f:
            json.dump(rename_dict, f, indent=2)

    def rename_samples_with_widgets_always(self, sample_names):
        """
        Display interactive text boxes for renaming sample names.
        """
        # Ensure directory and file exist
        json_dir = os.path.dirname(self.json_file_path)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        if not os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'w') as f:
                json.dump({}, f, indent=2)
            print(f"Created new JSON file at {self.json_file_path}")
        else:
            print(f"JSON file already exists at {self.json_file_path}")

        existing_map = self.load_rename_map(self.json_file_path) or {}
        text_boxes = []

        for old_nm in sample_names:
            default_new_name = existing_map.get(old_nm, old_nm)
            tb = widgets.Text(
                value=default_new_name,
                description=f"{old_nm}:",
                description_tooltip=old_nm,
                layout=widgets.Layout(width='1200px', description_width='400px')
            )
            text_boxes.append(tb)

        remove_prefix_button = widgets.Button(description="Shorten Names (Remove Common Prefix)")
        save_button = widgets.Button(description="Save Sample Names")
        output_area = widgets.Output()
        final_names = list(sample_names)

        def on_remove_prefix_clicked(_):
            with output_area:
                clear_output()
                current_vals = [tb.value for tb in text_boxes]
                prefix = os.path.commonprefix(current_vals)
                if prefix:
                    for tb in text_boxes:
                        tb.value = tb.value.replace(prefix, "", 1)
                    print(f"Removed common prefix: '{prefix}'")
                else:
                    print("No common prefix found among these names.")

        def on_save_button_clicked(_):
            with output_area:
                clear_output()
                rename_map = {}
                for old_nm, tb in zip(sample_names, text_boxes):
                    rename_map[old_nm] = tb.value
                self.save_rename_map(self.json_file_path, rename_map)
                for i, tb in enumerate(text_boxes):
                    final_names[i] = tb.value
                print(
                    f"Saved sample rename map to '{self.json_file_path}'.\n"
                    f"Your final names:\n{final_names}\n"
                    "Now click the 'Apply New Names & Rebuild Data' button below."
                )

        remove_prefix_button.on_click(on_remove_prefix_clicked)
        save_button.on_click(on_save_button_clicked)

        vbox = widgets.VBox([
            widgets.Label("Edit sample names below or remove common prefix, then click 'Save Sample Names':"),
            *text_boxes,
            remove_prefix_button,
            save_button,
            output_area
        ])
        display(vbox)
        return final_names

    def on_apply_button_clicked(self, _):
        """
        Event handler for the "Apply New Names & Rebuild Data" button.
        """
        with self.apply_output:
            clear_output()  # Clear previous messages

            # Re-load the JSON mapping and rebuild final names, using a default if needed.
            rename_map = self.load_rename_map(self.json_file_path) or {}
            final_names = [rename_map.get(old_nm, old_nm) for old_nm in self.sample_names_extracted]
            self.abundance_cols = final_names  # these are the new sample names

            # Rebuild the mass_spec_abundance DataFrame with updated names.
            accession_idx = self.column_letter_to_index(self.accession_col_letter)
            self.mass_spec_abundance = self.mass_spec_data.loc[
                :,
                [self.mass_spec_data.columns[accession_idx]] + self.area_norm_columns
            ].copy()
            self.mass_spec_abundance.columns = ["Accession"] + final_names
            self.mass_spec_abundance["Genes"] = self.genes_column
            self.mass_spec_abundance["Description"] = self.description_column

            # Re-merge with POI data.
            poi_data = pd.read_excel(self.poi_file_path)
            self.mass_spec_abundance["Accession"] = self.mass_spec_abundance["Accession"].astype(str)
            poi_data["Other UniProt Accessions"] = poi_data["Other UniProt Accessions"].astype(str)
            poi_data_exploded = poi_data.assign(
                **{"Other UniProt Accessions": poi_data["Other UniProt Accessions"].str.split(',')}
            ).explode("Other UniProt Accessions")
            poi_data_exploded["Other UniProt Accessions"] = poi_data_exploded["Other UniProt Accessions"].astype(str)

            def generate_color_palette(classes):
                pal = sns.color_palette("husl", len(classes))
                return {cls: mcolors.to_hex(pal[i]) for i, cls in enumerate(classes)}

            def load_color_map(filepath):
                try:
                    color_code_data = pd.read_csv(filepath)
                    return color_code_data.set_index("Class / family")["Color Hex"].to_dict()
                except FileNotFoundError:
                    print("Color code file not found. Generating colors automatically.")
                    return generate_color_palette(poi_data["Class / family"].unique())

            if self.color_code_file_path:
                cm = load_color_map(self.color_code_file_path)
            else:
                cm = generate_color_palette(poi_data["Class / family"].unique())

            self.color_map = cm

            poi_with_colors = pd.merge(
                poi_data_exploded,
                pd.DataFrame(list(self.color_map.items()), columns=["Class / family", "Color Hex"]),
                on="Class / family", how="left"
            )
            self.merged_data = pd.merge(
                self.mass_spec_abundance,
                poi_with_colors,
                how="left",
                left_on="Accession",
                right_on="Other UniProt Accessions"
            )

            print("\nRebuild complete. New column names are:")
            print(self.merged_data.columns.tolist())

    def run_preprocessing(self):
    """
    Execute the entire preprocessing workflow, including:
      - Preliminary directory checks (with a reminder)
      - Package installation
      - Mass spec data loading
      - Abundance column detection
      - Widget-based renaming
      - "Apply & Rebuild" button creation
    """
    # --- PRELIMINARY CHECKS ---
    # Remind the user where the file should be located (but do not interrupt processing)
    if not self.mass_spec_file_path.startswith(self.required_path):
        print("WARNING: The mass spec file should be stored in the shared directory:")
        print(self.required_path)
        print("Continuing processing...")

    # --- INSTALL REQUIRED PACKAGES ---
    required_packages = [
        'networkx', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'mplcursors',
        'mpld3', 'ipympl', 'kaleido', 'PyPDF2', 'statsmodels'
    ]
    self.install_packages(required_packages)

    # --- LOAD MASS SPEC DATA ---
    print("Loading data, please be patient...")
    self.mass_spec_data_full_headers = pd.read_excel(
        self.mass_spec_file_path, sheet_name='Main', header=None, nrows=5
    )

    # Identify the abundance columns
    row_for_labels = 3  # row index in the raw header lines
    area_norm_column_indices = [
        idx for idx, cell in enumerate(self.mass_spec_data_full_headers.iloc[row_for_labels])
        if any(ab_text in str(cell) for ab_text in self.abundance_column_label)
    ]

    # Extract sample names from the row above
    self.sample_names_extracted = [
        self.mass_spec_data_full_headers.iloc[row_for_labels - 1, idx]
        for idx in area_norm_column_indices
    ]

    # Load the main data with multi-level headers
    self.mass_spec_data = pd.read_excel(self.mass_spec_file_path, sheet_name='Main', header=[3, 4])

    # Get the columns by letter
    self.genes_column, genes_colname = self.get_column_by_letter(self.mass_spec_data, self.genes_col_letter)
    self.accession_column, accession_colname = self.get_column_by_letter(self.mass_spec_data, self.accession_col_letter)
    self.description_column, description_colname = self.get_column_by_letter(self.mass_spec_data, self.description_col_letter)

    print("\nPreview of the specified columns:")
    print(self.mass_spec_data[[genes_colname, accession_colname, description_colname]].head(10))
    print("\nPlease check carefully that the 'Genes', 'Accession', and 'Description' columns above are correct.")
    print("Proceeding automatically without user confirmation...\n")

    # --- DETECT ABUNDANCE COLUMNS ---
    self.area_norm_columns = [
        col for col in self.mass_spec_data.columns
        if any(ab_text in col[0] for ab_text in self.abundance_column_label)
    ]
    print("Detected abundance columns:")
    for col in self.area_norm_columns:
        print("   ", col)

    print("\nThese columns correspond to sample names:")
    for idx, (col, sname) in enumerate(zip(self.area_norm_columns, self.sample_names_extracted)):
        print(f"[{idx}] {col} -> {sname}")

    # --- WIDGET-BASED RENAMING ---
    print("\nLaunching the renaming widget UI...\n")
    self.rename_samples_with_widgets_always(self.sample_names_extracted)

    # --- APPLY THE NEW NAMES & REBUILD DATA ---
    self.apply_button.on_click(self.on_apply_button_clicked)
    display(self.apply_button, self.apply_output)
    print("Widget-based renaming logic initialized. Please rename samples")
    print(" and then click 'Save Sample Names', and DO NOT FORGET TO CLICK 'Apply New Names & Rebuild Data'.\n")




import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_overall_sample_composition(merged_data, abundance_cols, color_map):
    """
    Plot the overall sample composition for each sample in `abundance_cols`.
    This function:
      - Fills missing class/family names with 'Unknown', then recasts them as 'Non-POI'
      - Groups data by 'Class / family' and computes relative abundance
      - Plots the data on a log2 scale, labeling each bar with the number of unique proteins
    """
    # Ensure 'Class / family' is set to 'Non-POI' where missing
    merged_data['Class / family'] = merged_data['Class / family'].fillna('Unknown').replace('Unknown', 'Non-POI')

    # Group by 'Class / family' to count unique proteins
    unique_protein_counts = merged_data.groupby('Class / family')['Accession'].nunique().reset_index()
    unique_protein_counts.columns = ['Class / family', 'Unique Proteins']

    # Sum the abundance values for each class/family
    class_abundance = merged_data.groupby('Class / family')[abundance_cols].sum().reset_index()

    # Calculate relative abundance by dividing each abundance by the total across samples
    total_abundance = class_abundance[abundance_cols].sum()
    relative_abundance = class_abundance.copy()
    relative_abundance[abundance_cols] = relative_abundance[abundance_cols].div(total_abundance, axis=1)

    # Create subplots for each condition in abundance_cols
    fig, axes = plt.subplots(
        nrows=len(abundance_cols),
        figsize=(8, 6 * len(abundance_cols)),
        sharey=True
    )

    # If there is only one condition, axes is just a single Axes object
    if len(abundance_cols) == 1:
        axes = [axes]

    # Plot each condition separately
    for ax, condition in zip(axes, abundance_cols):
        sorted_data = relative_abundance.sort_values(by=condition, ascending=False)
        
        sns.barplot(
            x='Class / family',
            y=condition,
            data=sorted_data,
            ax=ax,
            hue='Class / family',       # Assign colors by class/family
            dodge=False,
            palette=[color_map.get(x, '#D3D3D3') for x in sorted_data['Class / family']],
            legend=False
        )
        ax.set_title(f'Relative Protein Class Abundance in {condition}')
        ax.set_xlabel('Protein Class')
        ax.set_ylabel('Relative Abundance (Log2 Scale, Powers of 10)')

        # Use log scale with base 2 for the y-axis
        ax.set_yscale('log', base=2)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2))
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda y, _: f'$10^{{{np.log10(y):.0f}}}$')
        )

        # Annotate bars with unique protein counts
        for bar, label in zip(ax.patches, sorted_data['Class / family']):
            unique_proteins = unique_protein_counts.loc[
                unique_protein_counts['Class / family'] == label,
                'Unique Proteins'
            ].values[0]
            ax.annotate(
                f' n= {unique_proteins} ',
                (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='right', va='bottom', fontsize=5, rotation=90
            )

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment('right')
            lbl.set_x(lbl.get_position()[0] + 0.05)

    plt.tight_layout()
    plt.show()

def display_class_condition_selection_widgets(merged_data, abundance_cols):
    """
    Display interactive widgets for class and condition selection.
    - Fills missing 'Class / family' with 'Unknown' -> 'Non-POI'
    - Displays multi-select widgets for classes and for conditions.
    - On button click, returns a dictionary with selected classes and conditions.
    """
    # Ensure proper values in the DataFrame
    merged_data['Class / family'] = merged_data['Class / family'].fillna('Unknown').replace('Unknown', 'Non-POI')
    
    # Calculate counts and build display strings
    class_counts = merged_data.groupby('Class / family')['Accession'].nunique().reset_index()
    class_counts.columns = ['Class / Family', 'Number of Proteins']
    class_counts = class_counts.sort_values(by='Number of Proteins', ascending=False)
    class_names_with_counts = [
        f"{name} ({count} proteins)" for name, count in 
        zip(class_counts['Class / Family'], class_counts['Number of Proteins'])
    ]
    class_name_map = dict(zip(class_names_with_counts, class_counts['Class / Family']))
    
    # Create the multi-select widgets
    class_selector = widgets.SelectMultiple(
        options=class_names_with_counts,
        description='Select Class(es)',
        rows=10,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    condition_selector = widgets.SelectMultiple(
        options=abundance_cols,
        description='Select Conditions',
        rows=6,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    
    # Display the widgets
    display(class_selector, condition_selector)
    
    # Prepare an output area and a dictionary to store selections.
    button_add = widgets.Button(description="Add classes/conditions to analysis")
    output = widgets.Output()
    selection_results = {"classes_to_plot": [], "conditions_to_plot": []}
    
    def on_button_click(b):
        with output:
            clear_output(wait=True)
            selected_classes = [class_name_map[s] for s in class_selector.value]
            selected_conditions = list(condition_selector.value)
            selection_results["classes_to_plot"] = selected_classes
            selection_results["conditions_to_plot"] = selected_conditions
            print("Selected classes added:")
            print(selected_classes)
            print("Selected conditions to plot:")
            print(selected_conditions)
            print("Re-run the ranked abundance cell to visualize the selected families.")
    
    button_add.on_click(on_button_click)
    display(button_add, output)
    
    return selection_results


# ------------------ New Class: InteractiveFieldSelector ------------------

class InteractiveFieldSelector:
    def __init__(self, data, default_hover_fields=None, exclude_fields=None):
        """
        Initialize the interactive field selector.
        
        Parameters:
          - data: A DataFrame containing your dataset.
          - default_hover_fields: A list of fields that should be selected by default.
          - exclude_fields: Fields (e.g. abundance columns) to exclude from the list.
        """
        self.data = data
        self.exclude_fields = exclude_fields or []
        self.available_fields = [col for col in data.columns.tolist() if col not in self.exclude_fields]
        self.selected_hover_fields = default_hover_fields or []
        self.selected_classes = []
        self._setup_widgets()

    def _setup_widgets(self):
        from ipywidgets import Checkbox, VBox, Button, Output, Label, SelectMultiple

        self.hover_config_output = Output()

        # Create a checkbox for each available field
        self.field_checkboxes = [
            Checkbox(value=(col in self.selected_hover_fields), description=col)
            for col in self.available_fields
        ]

        self.hover_submit_button = Button(
            description="Apply Hover Configuration",
            button_style="success"
        )
        self.hover_submit_button.on_click(self._apply_hover_config)

        # Class selection widget (if 'Class / family' exists in the data)
        if 'Class / family' in self.data.columns:
            class_vals = self.data['Class / family'].dropna().unique().tolist()
        else:
            class_vals = []
        self.class_selection = SelectMultiple(
            options=class_vals,
            description="Select Classes:",
            layout={"width": "300px", "height": "150px"}
        )
        self.class_selection.observe(self._on_class_selection_change, names='value')
        self.class_output = Output()

        # Build UI for hover fields and class selection
        self.hover_box = VBox([
            Label("Select fields to include in hover text:"),
            VBox(self.field_checkboxes, layout={"max_height": "300px", "overflow": "auto"}),
            self.hover_submit_button,
            self.hover_config_output
        ])
        self.class_box = VBox([
            Label("Select classes to highlight:"),
            self.class_selection,
            self.class_output
        ])
        self.interface = VBox([self.hover_box, self.class_box])

    def _apply_hover_config(self, b):
        self.selected_hover_fields = [cb.description for cb in self.field_checkboxes if cb.value]
        with self.hover_config_output:
            self.hover_config_output.clear_output()
            print("Selected hover fields:", self.selected_hover_fields)

    def _on_class_selection_change(self, change):
        self.selected_classes = list(change['new'])
        with self.class_output:
            self.class_output.clear_output()
            print("Selected classes:", self.selected_classes)

    def get_selected_hover_fields(self):
        return self.selected_hover_fields

    def get_selected_classes(self):
        return self.selected_classes

    def display_full(self):
        from IPython.display import display
        display(self.interface)

    def display_hover_only(self):
        from IPython.display import display
        display(self.hover_box)

    def reset(self):
        for cb in self.field_checkboxes:
            cb.value = False
        self.selected_hover_fields = []
        self.selected_classes = []
        self.class_selection.value = ()
        with self.hover_config_output:
            self.hover_config_output.clear_output()
        with self.class_output:
            self.class_output.clear_output()
            
            
            
#ranked abundance plot:

# ------------------ HELPER FUNCTIONS FOR RANKED ABUNDANCE ------------------

def sanitize_filename(filename):
    """Make a filename safe for most filesystems."""
    return "".join([c if c.isalnum() or c in [' ', '_', '-'] else "_" for c in filename]).strip()


def generate_color_palette(classes):
    """
    Generate a dictionary mapping each class/family to a distinct color using seaborn's 'husl' palette.
    """
    palette = sns.color_palette("husl", len(classes))
    color_map = {}
    for cls, (r, g, b) in zip(classes, palette):
        color_map[cls] = '#{:02x}{:02x}{:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
    return color_map


def create_ranked_abundance_figure_all_classes(data, condition, classes_to_highlight, color_map, add_labels_for_pdf=False, hover_fields=None):
    sorted_data = data.sort_values(by=condition, ascending=False).reset_index(drop=True)
    col_log_name = f"log10_{condition}"
    sorted_data[col_log_name] = np.log10(sorted_data[condition] + 1)
    # Use default fields if none provided
    if hover_fields is None:
        hover_fields = ['Genes', 'Description']
    # Build hover text for all proteins dynamically:
    hover_text_bg = sorted_data[hover_fields].astype(str).apply(lambda row: "<br>".join(row), axis=1)
    fig_widget = go.FigureWidget()
    fig_widget.add_scatter(
        x=sorted_data.index,
        y=sorted_data[col_log_name],
        mode='markers',
        marker=dict(color='#D3D3D3', size=5, opacity=0.5),
        name='All Proteins (BG)',
        hovertext=hover_text_bg,
        hoverinfo='text'
    )
    for cls in classes_to_highlight:
        subset = sorted_data[sorted_data['Class / family'] == cls]
        if not subset.empty:
            class_color = color_map.get(cls, '#D3D3D3')
            trace_mode = 'markers+text' if add_labels_for_pdf else 'markers'
            hover_text_hl = subset[hover_fields].astype(str).apply(lambda row: "<br>".join(row), axis=1)
            fig_widget.add_scatter(
                x=subset.index,
                y=subset[col_log_name],
                mode=trace_mode,
                marker=dict(
                    color=class_color,
                    size=10,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=cls,
                text=subset['Genes'] if add_labels_for_pdf else None,
                textposition='top center',
                hovertext=hover_text_hl,
                hoverinfo='text'
            )
    fig_widget.update_layout(
        title=f"Ranked Abundance: {condition} (All Highlighted Classes)",
        xaxis_title='Protein Rank (descending abundance)',
        yaxis_title='log10(Abundance + 1)',
        legend_title='Classes (double click on one to isolate)',
        template='plotly_white',
        height=600,
        width=900
    )
    return fig_widget, sorted_data, col_log_name



def show_ranked_abundance_plot(fig_widget, sorted_data, col_log_name, condition, output_directory, timestamp, default_threshold=2.0):
    """
    Displays a FigureWidget with interactive widgets:
      - Threshold for log10(abundance).
      - Text input for filename suffix.
      - Checkboxes for each trace.
      - Buttons to toggle all traces, save plot, export hits.
      - A search box to highlight proteins by partial match in Genes/Accession.
      - A clear highlights button to remove all search highlights.
    """
    threshold_input = widgets.FloatText(
        value=default_threshold,
        description="Log10 Abundance Cutoff:",
        style={'description_width': '200px'},
        layout={'width': '300px'}
    )
    suffix_input = widgets.Text(
        value='',
        description="Suffix to add to file name:",
        style={'description_width': '300px'},
        layout={'width': '450px'}
    )
    trace_checkboxes = [
        widgets.Checkbox(value=True, description=tr.name, indent=False, layout={'width': '450px'})
        for tr in fig_widget.data
    ]
    toggle_all_button = widgets.Button(description="Toggle All Traces")
    save_plot_button = widgets.Button(description="Save Plot")
    export_hits_button = widgets.Button(description="Export Hits above threshold")
    search_input = widgets.Text(
        value='',
        description="Search Protein:",
        style={'description_width': '300px'},
        layout={'width': '450px'}
    )
    search_button = widgets.Button(description="Search")
    clear_button = widgets.Button(description="Clear Highlights")
    console_output = widgets.Output()
    display(fig_widget)
    display(widgets.Label("Click checkboxes below to include/exclude those traces in the saved figure:"))
    half = (len(trace_checkboxes) + 1) // 2
    cb_col1 = widgets.VBox(trace_checkboxes[:half])
    cb_col2 = widgets.VBox(trace_checkboxes[half:])
    row0 = widgets.HBox([toggle_all_button])
    row1 = widgets.HBox([cb_col1, cb_col2])
    row2b = widgets.HBox([threshold_input, export_hits_button])
    row3 = widgets.HBox([suffix_input, save_plot_button])
    row_search = widgets.HBox([search_input, search_button, clear_button])
    display(widgets.VBox([row0, row1, row2b, row3, row_search, console_output]))
    
    def get_file_suffix():
        raw_suffix = suffix_input.value.strip()
        if raw_suffix == "":
            return ""
        safe_suffix = sanitize_filename(raw_suffix)
        return f"_{safe_suffix}"
    
    def get_hits_above_threshold(th):
        return sorted_data[sorted_data[col_log_name] >= th]
    
    @console_output.capture(clear_output=True)
    def on_toggle_all_clicked(_):
        any_unchecked = any(not cb.value for cb in trace_checkboxes)
        new_val = True if any_unchecked else False
        for cb in trace_checkboxes:
            cb.value = new_val
    toggle_all_button.on_click(on_toggle_all_clicked)
    
    @console_output.capture(clear_output=True)
    def on_save_plot_clicked(_):
        original_visibility = [tr.visible for tr in fig_widget.data]
        for tr, cb in zip(fig_widget.data, trace_checkboxes):
            tr.visible = True if cb.value else 'legendonly'
        suffix_str = get_file_suffix()
        pdf_filename = os.path.join(
            output_directory,
            f"ranked_abundance_{condition}{suffix_str}_{timestamp}.pdf"
        )
        fig_widget.write_image(pdf_filename, format='pdf')
        html_filename = os.path.join(
            output_directory,
            f"ranked_abundance_{condition}{suffix_str}_{timestamp}.html"
        )
        fig_widget.write_html(html_filename)
        print(f"Plot saved to:\n  PDF: {pdf_filename}\n  HTML: {html_filename}")
        for tr, vis in zip(fig_widget.data, original_visibility):
            tr.visible = vis
    save_plot_button.on_click(on_save_plot_clicked)
    
    @console_output.capture(clear_output=True)
    def on_export_hits_clicked(_):
        th = threshold_input.value
        hits_df = get_hits_above_threshold(th)
        suffix_str = get_file_suffix()
        hits_filename = os.path.join(
            output_directory,
            f"hits_ranked_abundance_{condition}_cutoff_{th}{suffix_str}_{timestamp}.csv"
        )
        hits_df.to_csv(hits_filename, index=False)
        print(f"Exported {len(hits_df)} hits above log10 threshold {th} to:\n  {hits_filename}")
    export_hits_button.on_click(on_export_hits_clicked)
    
    @console_output.capture(clear_output=True)
    def on_search_clicked(_):
        query = search_input.value.strip()
        if not query:
            print("Please enter a protein name or partial string in the search box.")
            return
        mask = (
            sorted_data['Genes'].str.contains(query, case=False, na=False) |
            sorted_data['Accession'].str.contains(query, case=False, na=False)
        )
        matched_data = sorted_data[mask]
        if matched_data.empty:
            print("Protein not found.")
        else:
            if len(matched_data) > 1:
                print(f"Warning: {len(matched_data)} matches found. Highlighting all.")
            fig_widget.add_scatter(
                x=matched_data.index,
                y=matched_data[col_log_name],
                mode='markers+text',
                name="Search Highlight",
                text=[query]*len(matched_data),
                textposition='top center',
                marker=dict(
                    color='rgba(0,0,0,0)',
                    size=12,
                    line=dict(width=4, color='black')
                ),
                hovertext=(
                    matched_data['Genes'].fillna('Unknown') + "<br>" +
                    matched_data['Accession'].fillna('Unknown')
                ),
                hoverinfo='text'
            )
            print(f"Found {len(matched_data)} match(es). Highlight added.")
    search_button.on_click(on_search_clicked)
    
    @console_output.capture(clear_output=True)
    def on_clear_clicked(_):
        indices_to_remove = [i for i, tr in enumerate(fig_widget.data) if tr.name == "Search Highlight"]
        for idx in reversed(indices_to_remove):
            fig_widget.data = fig_widget.data[:idx] + fig_widget.data[idx+1:]
        print("All search highlights cleared.")
    clear_button.on_click(on_clear_clicked)


def set_default_output_directory(mass_spec_file_path):
    """Returns a default output directory based on the mass spec file's location."""
    return os.path.join(os.path.dirname(mass_spec_file_path), 'analysis')


def run_ranked_abundance_analysis(merged_data, conditions_to_plot, classes_to_plot, color_map, mass_spec_file_path, output_directory_input="", hover_fields=None):
    if output_directory_input:
        output_directory = output_directory_input
    else:
        output_directory = set_default_output_directory(mass_spec_file_path)
    os.makedirs(output_directory, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for condition in conditions_to_plot:
        fig_widget, sorted_data, col_log_name = create_ranked_abundance_figure_all_classes(
            data=merged_data,
            condition=condition,
            classes_to_highlight=classes_to_plot,
            color_map=color_map,
            add_labels_for_pdf=False,
            hover_fields=hover_fields  # Pass the selected hover fields if provided
        )
        show_ranked_abundance_plot(
            fig_widget=fig_widget,
            sorted_data=sorted_data,
            col_log_name=col_log_name,
            condition=condition,
            output_directory=output_directory,
            timestamp=timestamp,
            default_threshold=2.0
        )
    print("\nAll done!")

    
    
    
# --- LOG2-FOLD CHANGE ANALYSIS FUNCTIONS ---

def normalize_data(data, reference_protein_accession, conditions):
    """
    Normalizes data by dividing each condition's abundance by that of a reference protein.
    Creates new columns: '<cond>_normalized_<ref>'.
    """
    ref_data = data[data['Accession'] == reference_protein_accession]
    normalized_data = data.copy()
    for cond in conditions:
        ref_val = ref_data[cond].values[0]
        norm_col = f"{cond}_normalized_{reference_protein_accession}"
        normalized_data[norm_col] = data[cond] / ref_val
    print(f"Data normalized using {reference_protein_accession} as reference.")
    return normalized_data


def prepare_log2_fold_changes(data, conditions_to_plot, reference_protein=None, use_normalized=True):
    """
    Adds columns like '<condA> vs <condB>' to 'data', storing log2(condA/condB).
    If 'use_normalized' is True and 'reference_protein' is provided,
    it uses the normalized columns (<cond>_normalized_<reference_protein>).
    """
    log2_cols = {}
    if use_normalized and reference_protein:
        conds = [f"{c}_normalized_{reference_protein}" for c in conditions_to_plot]
    else:
        conds = conditions_to_plot

    from itertools import combinations
    for c1, c2 in combinations(conds, 2):
        col_name = f"{c1} vs {c2}"
        ratio = (
            data[c1].replace([0, np.inf, -np.inf], np.nan) /
            data[c2].replace([0, np.inf, -np.inf], np.nan)
        )
        log2_cols[col_name] = np.log2(ratio)
    df_log2 = pd.DataFrame(log2_cols, index=data.index)
    return data.join(df_log2, rsuffix='_log2FC')

def create_log2_fc_figure_widget(
    data,
    cond1,
    cond2,
    classes_to_highlight,
    color_map,
    reference_protein=None,
    use_normalized=True,
    add_labels_for_pdf=False,
    hover_fields=None
):
    import plotly.graph_objects as go
    fig_widget = go.FigureWidget()
    if use_normalized and reference_protein:
        col1 = f"{cond1}_normalized_{reference_protein}"
        col2 = f"{cond2}_normalized_{reference_protein}"
    else:
        col1 = cond1
        col2 = cond2
    col_fc = f"{col1} vs {col2}"
    data['abs_fc'] = data[col_fc].abs()
    sorted_data = data.sort_values('abs_fc', ascending=False).drop(columns=['abs_fc'])
    sorted_data[col_fc] = -sorted_data[col_fc]
    sorted_data['Genes'] = sorted_data['Genes'].fillna('N/A')
    y_rounded = sorted_data[col_fc].round(1)
    counts = y_rounded.value_counts()
    sorted_data['point_density'] = y_rounded.map(counts)
    base_jitter, max_jitter = 0.005, 0.05
    np.random.seed(42)
    sorted_data['jitter_amount'] = base_jitter + ((sorted_data['point_density'] - 1) / (counts.max() - 1)) * (max_jitter - base_jitter)
    sorted_data['x_jitter'] = np.random.uniform(-1, 1, size=len(sorted_data)) * sorted_data['jitter_amount']
    # Use default hover fields if not provided
    if hover_fields is None:
        hover_fields = ['Genes', 'Description', 'Accession']
    # Build hover text for background trace
    hover_text_bg = sorted_data[hover_fields].astype(str).apply(lambda row: "<br>".join(row), axis=1)
    non_cls = sorted_data[~sorted_data['Class / family'].isin(classes_to_highlight)]
    fig_widget.add_scatter(
        x=non_cls['x_jitter'],
        y=non_cls[col_fc],
        mode='markers',
        marker=dict(color='#D3D3D3', size=6, opacity=0.4),
        name='Other Proteins',
        hovertext=hover_text_bg,
        hoverinfo='text'
    )
    for cls in classes_to_highlight:
        sub = sorted_data[sorted_data['Class / family'] == cls]
        if sub.empty:
            continue
        ccol = color_map.get(cls, '#D3D3D3')
        trace_mode = 'markers+text' if add_labels_for_pdf else 'markers'
        hover_text_hl = sub[hover_fields].astype(str).apply(lambda row: "<br>".join(row), axis=1)
        fig_widget.add_scatter(
            x=sub['x_jitter'],
            y=sub[col_fc],
            mode=trace_mode,
            marker=dict(
                color=ccol,
                size=10,
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name=cls,
            text=sub['Genes'] if add_labels_for_pdf else None,
            textposition='top center',
            hovertext=hover_text_hl,
            hoverinfo='text'
        )
    fig_widget.update_layout(
        title=f"Log2(FC): {cond2} vs {cond1}",
        xaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
        yaxis_title=f"{cond2} vs {cond1} (Log2 Fold Change)",
        legend_title='Classes (double click to isolate)',
        template='plotly_white',
        height=600,
        width=900
    )
    return fig_widget, sorted_data, col_fc

def show_log2_fc_plot(
    fig_widget,
    sorted_data,
    col_name,
    cond1,
    cond2,
    output_dir,
    timestamp,
    default_threshold=1.0
):
    """
    Displays a FigureWidget with interactive UI for log2 fold change analysis.
    Includes threshold and suffix inputs, toggle checkboxes, export, save plot,
    search/highlight, and clear highlights.
    """
    from ipywidgets import Button, HBox, VBox, Output, FloatText, Text, Checkbox, Label
    from IPython.display import display, clear_output

    console_output = Output()

    threshold_input = FloatText(
        value=default_threshold,
        description="Log2FC Cutoff:",
        style={'description_width': '150px'},
        layout={'width': '300px'}
    )
    suffix_input = Text(
        value='',
        description="File Name Suffix:",
        style={'description_width': '150px'},
        layout={'width': '300px'}
    )

    trace_checkboxes = [
        Checkbox(value=True, description=tr.name, indent=False, layout={'width': '300px'})
        for tr in fig_widget.data
    ]
    toggle_all_button = Button(description="Toggle All Traces")
    save_plot_button = Button(description="Save Plot")
    export_hits_button = Button(description="Export Hits")
    search_input = Text(
        value='',
        description="Search Protein:",
        style={'description_width': '120px'},
        layout={'width': '400px'}
    )
    search_button = Button(description="Search")
    clear_button = Button(description="Clear Highlights")

    display(fig_widget)
    display(Label("Toggle traces below for saving (PDF/HTML):"))
    half = (len(trace_checkboxes) + 1) // 2
    cb_col1 = VBox(trace_checkboxes[:half])
    cb_col2 = VBox(trace_checkboxes[half:])
    row0 = HBox([toggle_all_button])
    row1 = HBox([cb_col1, cb_col2])
    row2 = HBox([threshold_input, export_hits_button])
    row3 = HBox([suffix_input, save_plot_button])
    row_search = HBox([search_input, search_button, clear_button])
    display(VBox([row0, row1, row2, row3, row_search, console_output]))

    def sanitize_filename(fname):
        return "".join(c if c.isalnum() or c in [' ', '_', '-'] else "_" for c in fname).strip()

    def get_file_suffix():
        raw = suffix_input.value.strip()
        if not raw:
            return ""
        return f"_{sanitize_filename(raw)}"

    def hits_by_threshold(th):
        if th >= 0:
            return sorted_data[sorted_data[col_name] >= th]
        else:
            return sorted_data[sorted_data[col_name] <= th]

    @console_output.capture(clear_output=True)
    def on_toggle_all_clicked(_):
        any_unchecked = any(not cb.value for cb in trace_checkboxes)
        new_val = True if any_unchecked else False
        for cb in trace_checkboxes:
            cb.value = new_val

    toggle_all_button.on_click(on_toggle_all_clicked)

    @console_output.capture(clear_output=True)
    def on_save_plot_clicked(_):
        original_visibility = [tr.visible for tr in fig_widget.data]
        for tr, cb in zip(fig_widget.data, trace_checkboxes):
            tr.visible = True if cb.value else 'legendonly'
        suffix = get_file_suffix()
        pdf_name = os.path.join(
            output_dir,
            f"log2fc_{cond1}_vs_{cond2}{suffix}_{timestamp}.pdf"
        )
        fig_widget.write_image(pdf_name, format='pdf')
        html_name = os.path.join(
            output_dir,
            f"log2fc_{cond1}_vs_{cond2}{suffix}_{timestamp}.html"
        )
        fig_widget.write_html(html_name)
        print(f"Saved plot:\n  PDF: {pdf_name}\n  HTML: {html_name}")
        for tr, vis in zip(fig_widget.data, original_visibility):
            tr.visible = vis

    save_plot_button.on_click(on_save_plot_clicked)

    @console_output.capture(clear_output=True)
    def on_export_hits_clicked(_):
        th = threshold_input.value
        df_hits = hits_by_threshold(th)
        sign = "pos" if th >= 0 else "neg"
        suffix = get_file_suffix()
        csv_name = os.path.join(
            output_dir,
            f"hits_{cond1}_vs_{cond2}_{sign}_{abs(th)}{suffix}_{timestamp}.csv"
        )
        df_hits.to_csv(csv_name, index=False)
        print(f"Exported {len(df_hits)} hits at threshold {th} -> {csv_name}")

    export_hits_button.on_click(on_export_hits_clicked)

    @console_output.capture(clear_output=True)
    def on_search_clicked(_):
        query = search_input.value.strip()
        if not query:
            print("Please enter a partial search string.")
            return
        mask = (
            sorted_data['Genes'].str.contains(query, case=False, na=False) |
            sorted_data['Accession'].str.contains(query, case=False, na=False)
        )
        matched = sorted_data[mask]
        if matched.empty:
            print("Protein not found.")
            return
        if len(matched) > 1:
            print(f"Warning: {len(matched)} matches. Highlighting all.")
        fig_widget.add_scatter(
            x=matched['x_jitter'],
            y=matched[col_name],
            mode='markers+text',
            name="Search Highlight",
            text=[query]*len(matched),
            textposition='top center',
            marker=dict(
                color='rgba(0,0,0,0)',
                size=12,
                line=dict(width=4, color='black')
            ),
            hovertext=(
                matched['Genes'].fillna('Unknown') + "<br>" +
                matched['Accession'].fillna('')
            ),
            hoverinfo='text'
        )
        print(f"Found {len(matched)} match(es). Highlight(s) added.")

    search_button.on_click(on_search_clicked)

    @console_output.capture(clear_output=True)
    def on_clear_clicked(_):
        indices = [i for i, t in enumerate(fig_widget.data) if t.name == "Search Highlight"]
        for idx in reversed(indices):
            fig_widget.data = fig_widget.data[:idx] + fig_widget.data[idx+1:]
        print("All search highlights cleared.")

    clear_button.on_click(on_clear_clicked)


# ------------------- VIOLIN PLOT AND SAVING FUNCTIONS -------------------

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from datetime import datetime
from matplotlib.colors import to_hex
from ipywidgets import Button, HBox, VBox, Output, Text, Checkbox, Label
from IPython.display import display, clear_output
from PyPDF2 import PdfMerger  # optional, if you plan to merge PDFs later

def add_jitter(x, jitter_amount=0.05):
    return x + np.random.uniform(-jitter_amount, jitter_amount, size=len(x))

def ensure_normalization(data, reference_protein_accession, conditions):
    """
    For each condition, ensures a normalized column exists.
    Normalized columns are named: "{condition}_normalized_{reference_protein_accession}".
    """
    normalized_data = data.copy()
    for condition in conditions:
        normalized_col = f"{condition}_normalized_{reference_protein_accession}"
        if normalized_col not in data.columns:
            reference_data = data[data['Accession'] == reference_protein_accession]
            if not reference_data.empty:
                reference_value = reference_data[condition].values[0]
                normalized_data[normalized_col] = data[condition] / reference_value
                print(f"Normalized data for {condition} using reference protein {reference_protein_accession}.")
            else:
                raise ValueError(f"Reference protein {reference_protein_accession} not found in the data.")
    return normalized_data

def generate_color_palette(classes):
    """
    Generate a dictionary mapping each class/family to a distinct color using seaborn's 'husl' palette.
    """
    palette = sns.color_palette("husl", len(classes))
    return {cls: to_hex(palette[i]) for i, cls in enumerate(classes)}

def add_staggered_significance_annotations_plotly(fig, data, sample_names, p_values, annotation_height):
    # Adjust p-values using Bonferroni correction
    _, p_adj, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    for index, ((i, j), p_val) in enumerate(zip(combinations(range(len(sample_names)), 2), p_adj)):
        height = annotation_height * (1 + 0.1 * index)
        if p_val < 0.05:
            fig.add_shape(type="line",
                          x0=i, x1=j,
                          y0=height, y1=height,
                          line=dict(color="black", width=1.5))
            fig.add_annotation(x=(i+j)/2, y=height, text=f"{p_val:.1e}",
                               showarrow=False, yshift=10)
        else:
            fig.add_shape(type="line",
                          x0=i, x1=j,
                          y0=height, y1=height,
                          line=dict(color="grey", width=0.75, dash="dash"))
            fig.add_annotation(x=(i+j)/2, y=height, text="n.s.",
                               showarrow=False, yshift=10, font=dict(color="grey"))
    return fig

#old
def plot_class_abundance_plotly(class_name, data, conditions, color_map, 
                                reference_protein_accession=None, plot_normalized_data=False, 
                                label_genes=False, hover_fields=None):
    """
    Plots a violin plot for proteins in a given class.
    Builds hover text using the fields provided in hover_fields.
    """
    filtered_data = data[data['Class / family'] == class_name]
    # Use default hover fields if none provided
    if hover_fields is None:
        hover_fields = ['Genes', 'Description']
    condition_columns = []
    fig = go.Figure()
    y_max_values = []
    for i, condition in enumerate(conditions):
        condition_col = f"{condition}_normalized_{reference_protein_accession}" if (plot_normalized_data and reference_protein_accession) else condition
        condition_columns.append(condition_col)
        if condition_col not in filtered_data.columns:
            raise KeyError(f"Column {condition_col} not found in the data.")
        current_y_max = filtered_data[condition_col].max()
        y_max_values.append(current_y_max)
        # Add violin trace for each condition
        fig.add_trace(go.Violin(
            y=filtered_data[condition_col],
            x=[i] * len(filtered_data),
            name=condition_col,
            box_visible=True,
            meanline_visible=True,
            points="all",
            pointpos=0,
            opacity=0.6,
            fillcolor=color_map.get(condition, '#CCCCCC'),
            hoverinfo='skip',
            line=dict(color='rgba(0, 0, 0, 0)')
        ))
        # Build hover text from selected fields
        hover_text = filtered_data[hover_fields].astype(str).apply(lambda row: "<br>".join(row), axis=1)
        jittered_x = add_jitter(np.full(len(filtered_data), i), jitter_amount=0.05)
        fig.add_trace(go.Scatter(
            y=filtered_data[condition_col].replace([0], np.nan),
            x=jittered_x,
            mode='markers+text' if label_genes else 'markers',
            marker=dict(color='black', size=6, opacity=1, line=dict(width=1, color='white')),
            text=filtered_data['Genes'].fillna('N/A') if label_genes else None,
            textposition='top center' if label_genes else None,
            hovertext=hover_text,
            hoverinfo='text'
        ))
    y_max = max(y_max_values) if y_max_values else 1
    annotation_height = 1.15 * y_max
    title_suffix = f"(Normalized to {reference_protein_accession})" if (plot_normalized_data and reference_protein_accession) else ""
    fig.update_layout(
        title=f"Abundance of {class_name} Proteins Across Samples {title_suffix}",
        yaxis_title="Normalized Abundance" if (plot_normalized_data and reference_protein_accession) else "Abundance",
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(condition_columns))),
            ticktext=condition_columns,
            title="Sample Condition"
        ),
        yaxis=dict(range=[None, annotation_height * 1.1]),
        showlegend=False,
        template='plotly_white',
        height=600,
        width=1000,
    )
    return fig


def show_violin_plot_with_save_button(fig, class_name, output_dir, timestamp):
    """
    Displays the violin plot figure along with interactive widgets for trace toggling
    and saving the plot as PDF/HTML.
    """
    suffix_input = Text(
        value='',
        description="File Name Suffix:",
        style={'description_width': '150px'},
        layout={'width': '400px'}
    )
    trace_checkboxes = []
    for i, trace in enumerate(fig.data):
        trace_name = trace.name if trace.name is not None else f"Trace {i+1}"
        cb = Checkbox(
            value=True,
            description=trace_name,
            indent=False,
            layout={'width': '400px'}
        )
        trace_checkboxes.append(cb)
    toggle_all_button = Button(description="Toggle All Traces")
    save_plot_button = Button(description="Save Plot")
    console_output = Output()
    display(fig)
    display(Label("Select traces to include in the saved plot:"))
    half = (len(trace_checkboxes) + 1) // 2
    cb_col1 = VBox(trace_checkboxes[:half])
    cb_col2 = VBox(trace_checkboxes[half:])
    display(HBox([suffix_input, save_plot_button]))
    display(console_output)
    def sanitize_filename(filename):
        return "".join([c if c.isalnum() or c in [' ', '_', '-'] else "_" for c in filename]).strip()
    def get_file_suffix():
        raw_suffix = suffix_input.value.strip()
        return f"_{sanitize_filename(raw_suffix)}" if raw_suffix != "" else ""
    def on_toggle_all_clicked(_):
        any_unchecked = any(not cb.value for cb in trace_checkboxes)
        new_value = True if any_unchecked else False
        for cb in trace_checkboxes:
            cb.value = new_value
    toggle_all_button.on_click(on_toggle_all_clicked)
    def on_save_plot_clicked(_):
        with console_output:
            console_output.clear_output()
            original_visibility = [trace.visible for trace in fig.data]
            for trace, cb in zip(fig.data, trace_checkboxes):
                trace.visible = True if cb.value else 'legendonly'
            suffix_str = get_file_suffix()
            pdf_filename = os.path.join(output_dir, f"violin_{class_name}{suffix_str}_{timestamp}.pdf")
            html_filename = os.path.join(output_dir, f"violin_{class_name}{suffix_str}_{timestamp}.html")
            try:
                fig.write_image(pdf_filename, format='pdf')
                fig.write_html(html_filename)
                print(f"Plot saved as:\nPDF: {pdf_filename}\nHTML: {html_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            for trace, vis in zip(fig.data, original_visibility):
                trace.visible = vis
    toggle_all_button.on_click(on_toggle_all_clicked)
    save_plot_button.on_click(on_save_plot_clicked)


# ------------------- HEATMAP ANALYSIS WITH WIDGET-BASED SAVING -------------------
# ------------------- CLASS-LEVEL HEATMAP WITH DENDROGRAM, SIGNIFICANCE, & WIDGET-BASED SAVING -------------------
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from IPython.display import display, clear_output

# Helper: Format significance markers
def significance_marker(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def ensure_normalization(data, reference_protein_accession, conditions):
    """Ensure normalized columns exist for each condition."""
    normalized_data = data.copy()
    for condition in conditions:
        normalized_col = f"{condition}_normalized_{reference_protein_accession}"
        if normalized_col not in data.columns:
            reference_data = data[data['Accession'] == reference_protein_accession]
            if not reference_data.empty:
                reference_value = reference_data[condition].values[0]
                normalized_data[normalized_col] = data[condition] / reference_value
                print(f"Normalized {condition} using {reference_protein_accession}.")
            else:
                raise ValueError(f"Reference protein {reference_protein_accession} not found.")
    return normalized_data

def calculate_pairwise_log2_fold_changes(data, conditions_to_plot, classes_to_plot, 
                                           reference_protein_accession=None, use_normalized_data=True):
    """Calculate log2 fold changes for each pairwise combination of conditions for selected classes."""
    pairwise_log2_fc = {}
    if use_normalized_data and reference_protein_accession:
        conditions = [f"{cond}_normalized_{reference_protein_accession}" for cond in conditions_to_plot]
    else:
        conditions = conditions_to_plot

    data_filtered = data[data['Class / family'].isin(classes_to_plot)].copy()
    for cond1, cond2 in combinations(conditions, 2):
        col_name = f"{cond1} vs {cond2}"
        data_pair = data_filtered[[cond1, cond2]].replace([np.inf, -np.inf, 0], np.nan)
        if data_pair[cond1].isna().all() or data_pair[cond2].isna().all():
            print(f"Skipping {col_name} due to missing/invalid data.")
            continue
        data_filtered[col_name] = np.log2(data_pair[cond1] / data_pair[cond2])
        class_avg = data_filtered.groupby('Class / family')[col_name].mean()
        pairwise_log2_fc[col_name] = class_avg
    return pairwise_log2_fc

def plot_average_log2_fold_changes_widget(pairwise_log2_fc, color_scale, timestamp):
    """
    Generate horizontal bar plots (as FigureWidgets) of average log2 fold changes.
    The figure title is automatically set to the comparison string (e.g. "ConditionA vs ConditionB").
    Returns a dict mapping comparison names to FigureWidgets.
    """
    import plotly.graph_objects as go
    figures = {}
    for comparison, avg_log2_fc in pairwise_log2_fc.items():
        avg_log2_fc = avg_log2_fc.sort_values(ascending=False)
        max_abs_value = max(abs(avg_log2_fc.values))
        fixed_length = 0.1  # uniform bar length
        num_classes = len(avg_log2_fc)
        # Use the global parameters BAR_HEIGHT_PER_CLASS and ADDITIONAL_MARGIN (set externally)
        height_for_plot = BAR_HEIGHT_PER_CLASS * num_classes + ADDITIONAL_MARGIN

        fig = go.FigureWidget()
        fig.add_trace(go.Bar(
            y=avg_log2_fc.index,
            x=[fixed_length] * num_classes,
            orientation='h',
            marker=dict(
                color=avg_log2_fc.values,
                colorscale=color_scale,
                cmin=-max_abs_value,
                cmax=max_abs_value,
                colorbar=dict(
                    title="Log2 Fold Change",
                    tickvals=[-max_abs_value, 0, max_abs_value],
                    ticktext=[f"-{max_abs_value:.2f}", "0", f"{max_abs_value:.2f}"]
                ),
            ),
            text=avg_log2_fc.round(2),
            textposition='outside',
            hovertext=[f"{value:.2f}" for value in avg_log2_fc.values],
            hoverinfo="text"
        ))
        fig.update_layout(
            title=f"<b>{comparison}</b>",
            xaxis=dict(title="", showgrid=False, zeroline=False, range=[0, 1], showticklabels=False),
            yaxis=dict(title="Protein Classes", autorange="reversed", tickfont=dict(size=10)),
            template="plotly_white",
            height=height_for_plot,
            width=FIG_WIDTH
        )
        figures[comparison] = fig
    return figures

def show_heatmap_plot_with_save_button(fig, comparison, html_dir, pdf_dir, timestamp, heatmap_dir):
    """
    Displays the given heatmap FigureWidget with widget controls to enter a file name suffix
    and a single "Save Plot" button. The suffix is appended to the file names.
    (No "toggle all" button is used.)
    """
    from ipywidgets import Button, HBox, VBox, Output, Text, Label
    from IPython.display import display
    # Convert the figure to a FigureWidget if not already one
    fig = go.FigureWidget(fig)
    suffix_input = Text(
        value='',
        description="File Name Suffix:",
        style={'description_width': '150px'},
        layout={'width': '400px'}
    )
    save_plot_button = Button(description="Save Plot")
    save_output = Output()
    container = VBox(children=[fig, HBox(children=[suffix_input, save_plot_button]), save_output])
    display(container)

    def sanitize_filename(fname):
        return "".join([c if c.isalnum() or c in [' ', '_', '-'] else "_" for c in fname]).strip()
    def get_file_suffix():
        raw = suffix_input.value.strip()
        return f"_{sanitize_filename(raw)}" if raw != "" else ""
    def on_save_plot_clicked(_):
        with save_output:
            save_output.clear_output()
            suffix_str = get_file_suffix()
            html_filename = os.path.join(heatmap_dir, f"{comparison}_{timestamp}{suffix_str}.html")
            pdf_filename = os.path.join(heatmap_dir, f"{comparison}_{timestamp}{suffix_str}.pdf")
            try:
                fig.write_html(html_filename)
                fig.write_image(pdf_filename, format="pdf")
                print(f"Saved heatmap:\nHTML: {html_filename}\nPDF: {pdf_filename}")
            except Exception as e:
                print(f"Error saving heatmap: {e}")
    save_plot_button.on_click(on_save_plot_clicked)

def run_heatmap_analysis_with_widgets(merged_data, conditions_to_plot, classes_to_plot, reference_protein_accession,
                                      plot_normalized_data, mass_spec_file_path, output_directory_input="",
                                      bar_height_per_class=80, additional_margin=400, fig_width=200):
    """
    Runs the heatmap analysis with widget-based saving:
      - Sets up output directories,
      - Optionally normalizes data,
      - Calculates pairwise log2 fold changes,
      - Generates bar plots (as FigureWidgets) of class-average log2FC for each comparison,
      - Displays each plot with its own interactive saving controls.
      
    The plot layout parameters (bar height, additional margin, figure width) are provided as arguments.
    The title of each plot is automatically generated from the comparison (e.g. "ConditionA vs ConditionB").
    """
    global BAR_HEIGHT_PER_CLASS, ADDITIONAL_MARGIN, FIG_WIDTH
    BAR_HEIGHT_PER_CLASS = bar_height_per_class
    ADDITIONAL_MARGIN = additional_margin
    FIG_WIDTH = fig_width

    output_directory = output_directory_input or os.path.join(os.path.dirname(mass_spec_file_path), 'analysis')
    os.makedirs(output_directory, exist_ok=True)
    heatmap_dir = os.path.join(output_directory, "heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)
    html_dir = os.path.join(heatmap_dir, "plots_html")
    pdf_dir = os.path.join(heatmap_dir, "plots_pdf")
    os.makedirs(html_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if plot_normalized_data and reference_protein_accession:
        merged_data = ensure_normalization(merged_data, reference_protein_accession, conditions_to_plot)
    else:
        print("Normalization turned off or no reference provided; using non-normalized data.")
    
    pairwise_log2_fc = calculate_pairwise_log2_fold_changes(
        merged_data,
        conditions_to_plot,
        classes_to_plot,
        reference_protein_accession=reference_protein_accession,
        use_normalized_data=plot_normalized_data
    )
    color_scale = ['#ffb6c1', '#ffffff', '#404080']
    figures = plot_average_log2_fold_changes_widget(pairwise_log2_fc, color_scale, timestamp)
    for comparison, fig in figures.items():
        print(f"Interactive saving for comparison: {comparison}")
        show_heatmap_plot_with_save_button(fig, comparison, html_dir, pdf_dir, timestamp, heatmap_dir)
    print("\nDone! Files (if saved) will appear under:", heatmap_dir)

# ------------------- CLASS-LEVEL HEATMAP WITH DENDROGRAM, SIGNIFICANCE, & WIDGET-BASED SAVING -------------------
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from IPython.display import display, clear_output

# Constants for heatmap layout
BAR_HEIGHT_PER_CLASS = 20   # vertical space per class for annotations
ADDITIONAL_MARGIN = 200     # extra margin for titles, labels, etc.
FIG_WIDTH = 1000            # fixed width for the heatmap

#set timetstamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Helper function to format significance markers
def significance_marker(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

def ensure_normalization(data, reference_protein_accession, conditions):
    """Ensure normalized columns exist for each condition."""
    normalized_data = data.copy()
    for condition in conditions:
        normalized_col = f"{condition}_normalized_{reference_protein_accession}"
        if normalized_col not in data.columns:
            reference_data = data[data['Accession'] == reference_protein_accession]
            if not reference_data.empty:
                reference_value = reference_data[condition].values[0]
                normalized_data[normalized_col] = data[condition] / reference_value
                print(f"Normalized {condition} using {reference_protein_accession}.")
            else:
                raise ValueError(f"Reference protein {reference_protein_accession} not found.")
    return normalized_data

def run_class_level_heatmap_analysis_with_dendrogram_and_widgets(
    merged_data,
    conditions_to_plot,
    classes_to_plot,
    reference_protein_accession,
    plot_normalized_data,
    mass_spec_file_path,
    output_directory_input="",
    perform_clustering=True,
    comparison_label="Class_Level_Heatmap",
    hover_fields_for_heatmap=None,
    bar_height_per_class=200,      # new: vertical space per class for annotations
    additional_margin=200,         # new: extra margin for titles, labels, etc.
    fig_width=10                   # new: fixed width for the heatmap (in your preferred units)
):
    """
    Performs class-level heatmap analysis with hierarchical clustering (dendrograms),
    significance testing between conditions, and displays the heatmap with widget-based
    saving controls (one save button per plot).

    The plot layout parameters can be overridden via the last three arguments.
    """
    # 1. Setup output directories
    output_directory = output_directory_input or os.path.join(os.path.dirname(mass_spec_file_path), 'analysis')
    os.makedirs(output_directory, exist_ok=True)
    heatmap_dir = os.path.join(output_directory, "class_level_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # 2. Normalize data if requested
    if plot_normalized_data and reference_protein_accession:
        data_norm = ensure_normalization(merged_data, reference_protein_accession, conditions_to_plot)
    else:
        data_norm = merged_data.copy()
    
    # 3. Define condition column names
    if plot_normalized_data:
        condition_cols = [f"{cond}_normalized_{reference_protein_accession}" for cond in conditions_to_plot]
    else:
        condition_cols = conditions_to_plot
    missing_cols = [col for col in condition_cols if col not in data_norm.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # 4. Filter data for selected classes and columns
    filtered_data = data_norm[data_norm['Class / family'].isin(classes_to_plot)]
    filtered_data = filtered_data[['Class / family'] + condition_cols]
    
    # 5. Aggregate at class level
    class_abundance_data = filtered_data.groupby('Class / family')[condition_cols].sum()
    
    # 6. Optionally perform clustering
    if perform_clustering:
        row_link = linkage(class_abundance_data, method='ward')
        col_link = linkage(class_abundance_data.T, method='ward')
        row_order = leaves_list(row_link)
        col_order = leaves_list(col_link)
        clustered_data = class_abundance_data.iloc[row_order, col_order]
        clustered_condition_cols = [condition_cols[i] for i in col_order]
    else:
        clustered_data = class_abundance_data
        clustered_condition_cols = condition_cols
    
    # 7. Standardize data (row-wise Z-scores)
    standardized_data = clustered_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1).fillna(0)
    
    # 8. Perform pairwise significance tests (Mann-Whitney U)
    sig_df = pd.DataFrame(index=clustered_data.index, 
                          columns=[f"{c1} vs {c2}" for c1, c2 in combinations(clustered_condition_cols, 2)])
    p_values_df = pd.DataFrame(index=clustered_data.index, columns=clustered_condition_cols)
    fold_changes_df = pd.DataFrame(index=clustered_data.index, 
                                   columns=[f"{c1} vs {c2}" for c1, c2 in combinations(clustered_condition_cols, 2)])
    significant_count = 0
    non_significant_count = 0
    for cls in clustered_data.index:
        for c1, c2 in combinations(clustered_condition_cols, 2):
            group1 = filtered_data.loc[filtered_data['Class / family'] == cls, c1].dropna()
            group2 = filtered_data.loc[filtered_data['Class / family'] == cls, c2].dropna()
            if not group1.empty and not group2.empty:
                _, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                marker = significance_marker(p_val)
                fc = group1.median() / group2.median() if group2.median() != 0 else np.inf
                log2_fc = np.log2(fc) if fc > 0 else np.nan
                fold_changes_df.at[cls, f"{c1} vs {c2}"] = log2_fc
                sig_df.at[cls, f"{c1} vs {c2}"] = marker
                if marker != 'ns':
                    significant_count += 1
                else:
                    non_significant_count += 1
                p_values_df.at[cls, c1] = p_val
                p_values_df.at[cls, c2] = p_val
    print(f"Significant changes: {significant_count}, Non-significant changes: {non_significant_count}")
    for cls in sig_df.index:
        for comp in sig_df.columns:
            if sig_df.at[cls, comp] != 'ns':
                c1, c2 = comp.split(" vs ")
                p_val = p_values_df.at[cls, c1]
                fc = fold_changes_df.at[cls, comp]
                sig_df.at[cls, comp] = f"{sig_df.at[cls, comp]} p={p_val:.2e}, FC={fc:.2f}"
    
    # 9. Prepare hover text for the heatmap
    if hover_fields_for_heatmap is None:
        hover_fields_for_heatmap = ['Genes', 'Description']
    hover_text = [[f"Class: {row}<br>Condition: {col}<br>Z-score: {standardized_data.at[row, col]:.2f}"
                    for col in clustered_condition_cols] for row in standardized_data.index]
    
    # 10. Create dendrograms if clustering is performed
    if perform_clustering:
        col_dendro = dendrogram(col_link, orientation='top', no_plot=True)
        row_dendro = dendrogram(row_link, orientation='left', no_plot=True)
    else:
        col_dendro = None
        row_dendro = None

    # 11. Create a subplot layout with dendrograms and heatmap as a FigureWidget
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.2, 0.8],
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.05,
        vertical_spacing=0.02,
        specs=[[{"type": "scatter"}, None],
               [{"type": "heatmap"}, {"type": "scatter"}]]
    )
    if col_dendro:
        for icoord, dcoord in zip(col_dendro['icoord'], col_dendro['dcoord']):
            fig.add_trace(
                go.Scatter(
                    x=[x - 0.5 for x in icoord],
                    y=dcoord,
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )
    heatmap = go.Heatmap(
        z=standardized_data.values,
        x=clustered_condition_cols,
        y=standardized_data.index,
        colorscale='Viridis',
        colorbar=dict(title='Z-score'),
        hoverinfo='text',
        text=hover_text,
        hoverongaps=False
    )
    fig.add_trace(heatmap, row=2, col=1)
    if row_dendro:
        for icoord, dcoord in zip(row_dendro['icoord'], row_dendro['dcoord']):
            fig.add_trace(
                go.Scatter(
                    x=dcoord,
                    y=[x - 0.5 for x in icoord],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                ),
                row=2, col=2
            )
    fig.update_layout(
        title='Class-Level Changes in Protein Abundance Across Selected Samples',
        xaxis=dict(domain=[0.0, 0.7]),
        xaxis2=dict(domain=[0.0, 0.7], tickangle=-45, title='Sample Conditions', automargin=True),
        yaxis=dict(domain=[0.8, 1.0], showticklabels=False),
        yaxis2=dict(title='Protein Classes', automargin=True, autorange='reversed'),
        template='plotly_white',
        height=800,
        width=fig_width,  # Use the passed fig_width parameter
        margin=dict(t=150, l=100, r=100, b=100)
    )
    # Convert to FigureWidget for widget compatibility
    fig = go.FigureWidget(fig)
    
    # 12. Widget-based saving controls (one save button per plot; no toggle-all)
    from ipywidgets import Button, HBox, VBox, Output, Text, Label
    container = VBox()
    suffix_input = Text(
        value='',
        description="Suffix:",
        style={'description_width': '150px'},
        layout={'width': '400px'}
    )
    save_plot_button = Button(description="Save Plot")
    save_output = Output()
    # Each figure gets its own container
    container.children = [fig, HBox([suffix_input, save_plot_button]), save_output]
    display(container)
    
    def sanitize_filename(fname):
        return "".join([c if c.isalnum() or c in [' ', '_', '-'] else "_" for c in fname]).strip()
    
    def get_file_suffix():
        raw = suffix_input.value.strip()
        return f"_{sanitize_filename(raw)}" if raw != "" else ""
    
    def on_save_plot_clicked(_):
        with save_output:
            save_output.clear_output()
            suffix_str = get_file_suffix()
            html_filename = os.path.join(heatmap_dir, f"{comparison_label}_{timestamp}{suffix_str}.html")
            pdf_filename = os.path.join(heatmap_dir, f"{comparison_label}_{timestamp}{suffix_str}.pdf")
            try:
                fig.write_html(html_filename)
                fig.write_image(pdf_filename, format="pdf")
                print(f"Saved heatmap:\nHTML: {html_filename}\nPDF: {pdf_filename}")
            except Exception as e:
                print(f"Error saving heatmap: {e}")
    save_plot_button.on_click(on_save_plot_clicked)
    
    # 13. Display significance summary table
    if not sig_df.empty:
        sig_df_styled = sig_df.style.set_caption("Significance Markers, P-Values, and Fold Changes").set_properties(**{'text-align': 'center'})
        display(sig_df_styled)
    
    print("\nDone! Heatmap and any saved files are located under:", heatmap_dir)
