import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict
from ipywidgets import (
    Dropdown, VBox, HBox, Output, Button, Label, Text, Checkbox, Layout,
    SelectMultiple, FloatText, Widget, widgets
)
from IPython.display import display

###############################################################################
# 1) DataLoader
###############################################################################
class DataLoader:
    def __init__(self, ms_data_path, poi_file_path, color_code_file_path, prefix="apQuant Area"):
        self.ms_data_path = ms_data_path
        self.poi_file_path = poi_file_path
        self.color_code_file_path = color_code_file_path
        self.prefix = prefix
        self.merged_data = None
        self.assignments_file = "group_assignments.json"
        self.group_assignments = defaultdict(list)
        self.color_mapping_dict = None  # Add this line

    def load_and_process_data(self, debug=False):
        # Load the mass spec data
        ms_data = pd.read_csv(self.ms_data_path, sep="\t")

        # Load the POI file and explode for multiple accessions
        poi_data = pd.read_excel(self.poi_file_path)
        poi_data_exploded = poi_data.assign(
            Other_UniProt_Accessions=poi_data['Other UniProt Accessions'].str.split(',')
        ).explode('Other_UniProt_Accessions')

        # Load the color code file
        color_code_data = pd.read_csv(self.color_code_file_path)

        # Create color mapping
        self.color_mapping_dict = color_code_data.set_index('Class / family')['Color Hex'].to_dict()
        if "Unlabeled" not in self.color_mapping_dict:
            self.color_mapping_dict["Unlabeled"] = "#808080"

        # Merge data
        self.merged_data = pd.merge(
            ms_data,
            poi_data_exploded,
            how='left',
            left_on='Accession',
            right_on='Other_UniProt_Accessions'
        )

        self.merged_data = pd.merge(
            self.merged_data,
            color_code_data,
            how='left',
            on='Class / family'
        )

        # Process merged data
        self.merged_data['Color Hex'] = self.merged_data['Color Hex'].fillna('#808080')
        self.merged_data['highlight'] = self.merged_data['Accession'].isin(
            poi_data_exploded['Other_UniProt_Accessions']
        )
        self.merged_data['Color'] = self.merged_data['Color Hex']

        # Print statistics
        self._print_statistics(poi_data_exploded)

        # Process quantification columns
        self.quantification_columns = [
            col for col in ms_data.columns if col.startswith(self.prefix)
        ]
        if debug:
            print(f"\nColumns starting with {self.prefix}:")
            for col in self.quantification_columns:
                print(col)

        return self.merged_data

    def _print_statistics(self, poi_data_exploded):
        num_proteins_poi = poi_data_exploded['Other_UniProt_Accessions'].nunique()
        num_classes_poi = poi_data_exploded['Class / family'].nunique()
        num_proteins_in_ms = self.merged_data['Accession'].nunique()
        num_classes_in_ms = self.merged_data['Class / family'].nunique()
        matched_proteins = self.merged_data[self.merged_data['Class / family'].notna()]
        num_pois_in_ms = matched_proteins['Accession'].nunique()

        print(f"Number of proteins in the mass spec file: {num_proteins_in_ms}")
        print(f"Number of POIs found in the mass spec file: {num_pois_in_ms}")
        print(f"Number of proteins in the POI file: {num_proteins_poi}")
        print(f"Number of Classes / families in the POI file: {num_classes_poi}")
        print(f"Number of Classes / families found in the mass spec file: {num_classes_in_ms}")


###############################################################################
# 2) InteractiveFieldSelector
###############################################################################
class InteractiveFieldSelector:
    def __init__(self, data, default_hover_fields=None):
        self.data = data
        self.available_fields = data.columns.tolist()
        self.selected_hover_fields = default_hover_fields or []
        self.selected_classes = []
        self._setup_widgets()

    def _setup_widgets(self):
        from ipywidgets import Checkbox, VBox, Button, Output, Label, SelectMultiple

        self.hover_config_output = Output()

        # Checkboxes for selecting hover fields
        self.field_checkboxes = [
            Checkbox(value=(col in self.selected_hover_fields), description=col)
            for col in self.available_fields
        ]

        self.hover_submit_button = Button(
            description="Apply Hover Configuration",
            button_style="success"
        )
        self.hover_submit_button.on_click(self._apply_hover_config)

        # Class selection
        class_vals = (
            self.data['Class / family'].dropna().unique()
            if 'Class / family' in self.data.columns else []
        )
        self.output = Output()
        self.class_selection = SelectMultiple(
            options=class_vals,
            description="Select Classes:",
            layout={"width": "300px", "height": "150px"}
        )
        self.class_selection.observe(self._on_class_selection_change, names='value')

        # Build the UI boxes
        self.hover_box = VBox([
            Label("Select fields to include in hover text:"),
            VBox(self.field_checkboxes, layout={"max_height": "300px", "overflow": "auto"}),
            self.hover_submit_button,
            self.hover_config_output
        ])
        self.class_box = VBox([
            Label("Select classes to highlight:"),
            self.class_selection,
            self.output
        ])
        self.interface = VBox([self.hover_box, self.class_box])

    def _apply_hover_config(self, b):
        self.selected_hover_fields = [
            cb.description for cb in self.field_checkboxes if cb.value
        ]
        with self.hover_config_output:
            self.hover_config_output.clear_output()
            print(f"Selected hover fields: {self.selected_hover_fields}")

    def _on_class_selection_change(self, change):
        self.selected_classes = list(change['new'])
        with self.output:
            self.output.clear_output()
            print(f"Selected classes: {self.selected_classes}")

    def get_selected_hover_fields(self):
        return self.selected_hover_fields

    def get_selected_classes(self):
        return self.selected_classes

    def display_full(self):
        display(self.interface)

    def display_hover_only(self):
        display(self.hover_box)

    def reset(self):
        for cb in self.field_checkboxes:
            cb.value = False
        self.selected_hover_fields = []
        self.selected_classes = []
        self.class_selection.value = ()
        with self.hover_config_output:
            self.hover_config_output.clear_output()
        with self.output:
            self.output.clear_output()


###############################################################################
# 3) VolcanoPlotManager with static label dropdown
###############################################################################

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ipywidgets as widgets
from ipywidgets import Layout, VBox, HBox, Button, Output, Text, Dropdown, Checkbox, Label
import plotly.graph_objects as go
from IPython.display import display, clear_output

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from ipywidgets import Layout, VBox, HBox, Button, Output, Text, Dropdown, Checkbox, Label
from IPython.display import display, clear_output

class VolcanoPlotManager:
    def __init__(
        self,
        plot_width=800,
        plot_height=600,
        show_threshold_lines=True,
        log2_fc_positive=2,
        log2_fc_negative=-2,
        p_value_threshold=0.05,
        unlabeled_dot_size=4,
        labeled_dot_size=8,
        unlabeled_opacity=0.5,
        labeled_opacity=1.0,
        color_mapping_dict=None,
        output_directory="volcano_plots",
        debug=False
    ):
        self.plot_settings = {
            'width': plot_width,
            'height': plot_height,
            'show_threshold_lines': show_threshold_lines,
            'log2_fc_positive': log2_fc_positive,
            'log2_fc_negative': log2_fc_negative,
            'p_value_threshold': p_value_threshold,
            'unlabeled_dot_size': unlabeled_dot_size,
            'labeled_dot_size': labeled_dot_size,
            'unlabeled_opacity': unlabeled_opacity,
            'labeled_opacity': labeled_opacity
        }
        self.color_mapping_dict = color_mapping_dict or {"Unlabeled": "#808080"}
        self.output_directory = output_directory
        self.debug = debug
        
        # For searching
        self.search_input = widgets.Text(
            description="Search Protein:",
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px')
        )
        self.search_button = widgets.Button(description="Search", button_style='primary')
        self.clear_search_button = widgets.Button(description="Clear Search", button_style='warning')
        self.search_button.on_click(self._on_search_clicked)
        self.clear_search_button.on_click(self._on_clear_search_clicked)

        # Data references
        self.result = None
        self.current_figure = None
        
        # Store selected row‐indices in a **set** to avoid duplicates
        self.selected_indices_set = set()
        
        # If you prefer single‐selection usage, see notes at end
        self.last_selection = None
        self.field_selector = None
        self.static_label_field_dropdown = None

        # Additional info
        self.current_log2fc_col = None
        self.current_pval_col = None

        os.makedirs(self.output_directory, exist_ok=True)
        self._setup_widgets()

    def set_field_selector(self, field_selector):
        self.field_selector = field_selector

    def _setup_widgets(self):
        self.plot_output = widgets.Output()
        self.message_output = widgets.Output()
        self.selection_output = widgets.Output()

        self.save_selection_button = widgets.Button(description='Save Selected Points', button_style='success')
        self.save_selection_button.on_click(self._save_selected_points)

        self.clear_selection_button = widgets.Button(description='Clear Selection', button_style='warning')
        self.clear_selection_button.on_click(self._clear_selection)

        self.save_plot_button = widgets.Button(description='Save Plot', button_style='info')
        self.save_plot_button.on_click(self._save_plot)

        self.filename_widget = widgets.Text(
            value="volcano_plot",
            description="File Name:",
            style={'description_width': 'initial'}
        )

        self.static_label_field_dropdown = widgets.Dropdown(
            options=[('None', None)],
            description='Static Labels:',
            value=None,
            style={'description_width': 'initial'}
        )

        self.layer_checkboxes = []
        self.layer_box = widgets.VBox([
            widgets.Label("Select layers to include in saved plot:"),
            widgets.VBox([], layout={"max_height": "300px", "overflow": "auto"})
        ])

    def create_volcano_plot(
        self,
        data,
        log2fc_col='log2FoldChange',
        pval_col='adjPValue',
        title="Volcano Plot",
        hover_fields=None
    ):
        fig = go.FigureWidget()
        self.current_figure = fig

        self.current_log2fc_col = log2fc_col
        self.current_pval_col = pval_col
        self.result = data.copy()

        try:
            df = data.copy()
            df['-log10(pValue)'] = -np.log10(df[pval_col].replace([0, np.nan], 1e-300))

            # Determine hover fields
            if not hover_fields and self.field_selector:
                hover_fields = self.field_selector.get_selected_hover_fields()
            if not hover_fields:
                hover_fields = ["Accession", "Description"]

            # Build hover text
            df['hover_text'] = df.apply(
                lambda row: "<br>".join(
                    f"{hf}: {row[hf]}" for hf in hover_fields if hf in row and pd.notnull(row[hf])
                ),
                axis=1
            )

            # 1) Base trace: "All Proteins (BG)"
            base_trace = go.Scatter(
                x=df[log2fc_col],
                y=df['-log10(pValue)'],
                mode='markers',
                marker=dict(
                    size=self.plot_settings['unlabeled_dot_size'],
                    color='lightgrey',
                    opacity=self.plot_settings['unlabeled_opacity'],
                    line=dict(color='black', width=0.5)
                ),
                name='All Proteins (BG)',
                text=df['hover_text'],
                hoverinfo='text',
                customdata=df.index,  # row indices
            )
            fig.add_trace(base_trace)

            # 2) Labeled classes
            if 'Class / family' in df.columns:
                labeled_data = df.dropna(subset=['Class / family'])
                if not labeled_data.empty:
                    for class_name, subdf in labeled_data.groupby('Class / family'):
                        trace_labeled = go.Scatter(
                            x=subdf[log2fc_col],
                            y=subdf['-log10(pValue)'],
                            mode='markers',
                            marker=dict(
                                size=self.plot_settings['labeled_dot_size'],
                                color=self.color_mapping_dict.get(class_name, '#808080'),
                                opacity=self.plot_settings['labeled_opacity'],
                                line=dict(color='black', width=0.5)
                            ),
                            name=str(class_name),
                            text=subdf['hover_text'],
                            hoverinfo='text',
                            customdata=subdf.index
                        )
                        fig.add_trace(trace_labeled)

            # Layout
            fig.update_layout(
                title=title,
                xaxis_title="Log2 Fold Change",
                yaxis_title="-Log10 Adjusted P-Value",
                template="plotly_white",
                width=self.plot_settings['width'],
                height=self.plot_settings['height'],
                showlegend=True,
                legend_title="Protein Classes",
                dragmode='select',
                selectdirection='any',
                clickmode='event+select'
            )

            # Threshold lines
            if self.plot_settings['show_threshold_lines']:
                fig.add_hline(
                    y=-np.log10(self.plot_settings['p_value_threshold']),
                    line_dash="dash",
                    line_color="grey",
                    opacity=0.5
                )
                fig.add_vline(
                    x=self.plot_settings['log2_fc_positive'],
                    line_dash="dash",
                    line_color="grey",
                    opacity=0.5
                )
                fig.add_vline(
                    x=self.plot_settings['log2_fc_negative'],
                    line_dash="dash",
                    line_color="grey",
                    opacity=0.5
                )

            # Attach selection. If you only want the base trace to be selectable, do:
            #   if trace.name == "All Proteins (BG)": ...
            # If you want labeled too, remove the condition.
            for trace in fig.data:
                # Let's attach selection to ALL traces so user can lasso from either base or labeled
                trace.on_selection(self._handle_selection)

            # Display
            self.plot_output.clear_output()
            with self.plot_output:
                display(fig)

        except Exception as e:
            print(f"Error in create_volcano_plot: {e}")

        return fig

    def _handle_selection(self, trace, points, selector):
        """
        Gather row indices from the selected event. We'll store them in a set
        so that duplicates can't accumulate if the same row is selected multiple
        times or from multiple layers.
        """
        newly_selected = set()
        if points.point_inds:
            # Convert local indices -> row indices from trace.customdata
            newly_selected = {trace.customdata[i] for i in points.point_inds}
        
        # Union with our existing set
        old_count = len(self.selected_indices_set)
        self.selected_indices_set = self.selected_indices_set.union(newly_selected)
        new_count = len(self.selected_indices_set)

        with self.selection_output:
            self.selection_output.clear_output()
            if self.selected_indices_set:
                added = new_count - old_count
                print(f"Currently selected rows (May contain duplicates that are removed upon saving): {len(self.selected_indices_set)}.")
                if added > 0:
                    print(f"Added {added} new row(s) to selection from this event. Use 'Save Selected Points' to export.")
                else:
                    print("No *new* unique rows were added from this selection.")
            else:
                print("No points selected (empty selection).")

    def _save_selected_points(self, b):
        """
        Export the currently selected row indices as CSV,
        removing duplicates if desired.
        """
        with self.selection_output:
            self.selection_output.clear_output()
            try:
                # Pull from the set we use in _handle_selection
                indices = self.selected_indices_set
                if not indices:
                    print("No points selected!")
                    return

                # Convert set -> list
                indices_list = list(indices)
                
                selected_data = self.result.loc[indices_list]

                # Optionally remove duplicates by Accession
                num_before = len(selected_data)
                selected_data = selected_data.drop_duplicates(subset=["Accession"])
                num_after = len(selected_data)

                out_file = os.path.join(self.output_directory, "selected_points.csv")
                selected_data.to_csv(out_file, index=True)

                if num_after < num_before:
                    removed = num_before - num_after
                    print(f"Warning: {removed} duplicate row(s) removed based on Accession.")
                print(f"Saved {len(selected_data)} total unique row(s) to: {out_file}")
            except Exception as e:
                print(f"Error saving points: {e}")




    def _clear_selection(self, b):
        """
        Clear the global set of selected row indices
        and also set selectedpoints=None in the figure's data.
        """
        self.selected_indices_set.clear()
        if self.current_figure:
            for tr in self.current_figure.data:
                tr.selectedpoints = None
        with self.selection_output:
            self.selection_output.clear_output()
            print("Selection cleared. (Global set is now empty.)")

    def _save_plot(self, b):
        """
        Filter out layers that are unchecked in layer_checkboxes and
        save the figure as HTML and SVG.
        """
        with self.message_output:
            self.message_output.clear_output()
            try:
                if self.current_figure is None:
                    print("No plot to save!")
                    return

                selected_layers = [cb.description for cb in self.layer_checkboxes if cb.value]
                filtered_fig = go.Figure()
                for tr in self.current_figure.data:
                    if tr.name in selected_layers:
                        filtered_fig.add_trace(tr)

                layout_copy = {
                    'title': self.current_figure.layout.title,
                    'xaxis_title': self.current_figure.layout.xaxis.title.text,
                    'yaxis_title': self.current_figure.layout.yaxis.title.text,
                    'template': 'plotly_white',
                    'width': self.plot_settings['width'],
                    'height': self.plot_settings['height'],
                    'showlegend': True,
                    'legend_title': "Protein Classes"
                }
                filtered_fig.update_layout(**layout_copy)

                lbl_col = self.static_label_field_dropdown.value
                if lbl_col is not None:
                    if lbl_col != 'None' and lbl_col in self.result.columns:
                        text_vals = self.result[lbl_col].fillna("").astype(str).values
                        for t in filtered_fig.data:
                            t.update(mode="markers+text", text=text_vals, textposition='top center')

                base_filename = self.filename_widget.value
                html_path = os.path.join(self.output_directory, f"{base_filename}.html")
                svg_path = os.path.join(self.output_directory, f"{base_filename}.svg")

                filtered_fig.write_html(html_path)
                filtered_fig.write_image(svg_path)

                print("Plot saved successfully!")
                print(f"HTML: {os.path.abspath(html_path)}")
                print(f"SVG:  {os.path.abspath(svg_path)}")
            except Exception as e:
                print(f"Error saving plot: {e}")

    def _on_search_clicked(self, b):
        """Creates a 'Search Highlight: <query>' trace with matched points (not selectable by default)."""
        query = self.search_input.value.strip()
        if not query:
            with self.selection_output:
                self.selection_output.clear_output()
                print("Please enter a search string.")
            return
        if self.current_figure is None or self.result is None:
            with self.selection_output:
                self.selection_output.clear_output()
                print("No figure or data available for searching.")
            return

        df = self.result.copy()
        for c in ["Accession", "Genes", "Description"]:
            if c not in df.columns:
                df[c] = ""

        mask = (
            df["Accession"].str.contains(query, case=False, na=False) |
            df["Genes"].str.contains(query, case=False, na=False) |
            df["Description"].str.contains(query, case=False, na=False)
        )
        matched_data = df[mask]
        if matched_data.empty:
            with self.selection_output:
                self.selection_output.clear_output()
                print(f"No matches found for '{query}'.")
            return

        matched_data['-log10(pValue)'] = -np.log10(matched_data[self.current_pval_col].replace([0, np.nan], 1e-300))
        x_vals = matched_data[self.current_log2fc_col]
        y_vals = matched_data['-log10(pValue)']

        highlight_trace = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            name=f"Search Highlight: {query}",
            text=[query] * len(matched_data),
            textposition='top center',
            marker=dict(
                color='rgba(255,0,0,0.0)',
                size=12,
                line=dict(width=4, color='red')
            ),
            hovertext=matched_data.get('hover_text', None),
            hoverinfo='text',
            customdata=matched_data.index
        )
        if self.current_figure:
            self.current_figure.add_trace(highlight_trace)
            # If you want search highlights to be selectable:
            # highlight_trace.on_selection(self._handle_selection)

        with self.selection_output:
            self.selection_output.clear_output()
            print(f"Found {len(matched_data)} match(es). Highlight(s) added for '{query}'.")

    def _on_clear_search_clicked(self, b):
        """Removes all 'Search Highlight: ...' traces from the figure."""
        if self.current_figure is None:
            return
        new_traces = tuple(tr for tr in self.current_figure.data if not tr.name.startswith("Search Highlight: "))
        self.current_figure.data = new_traces
        with self.selection_output:
            self.selection_output.clear_output()
            print("All search highlights cleared.")

    def display_interface(self):
        """
        Show the main UI, which includes:
          - File name and static label fields
          - Buttons to save selection, clear selection, save plot
          - The search row
          - The layer checkbox box, message output, selection output
        """
        self.full_controls = VBox([
            HBox([self.filename_widget, self.static_label_field_dropdown]),
            HBox([
                self.save_plot_button,
                self.save_selection_button,
                self.clear_selection_button
            ], layout=Layout(margin='10px 0px')),
            HBox([self.search_input, self.search_button, self.clear_search_button])
        ])

        self.main_display = VBox([
            self.plot_output,
            VBox([self.full_controls, self.layer_box]),
            self.message_output,
            self.selection_output
        ])
        display(self.main_display)

###############################################################################
# 4) Filtering helper
###############################################################################
def filter_columns_by_keywords(dataframe, keywords):
    return [
        col for col in dataframe.columns
        if any(keyword.lower() in col.lower() for keyword in keywords)
    ]

###############################################################################
# 5) HeatmapManager
###############################################################################
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import (
    Dropdown, VBox, HBox, Output, Button, Label, Text, Checkbox, Layout,
    SelectMultiple, FloatText, Widget, widgets
)
from IPython.display import display

class HeatmapManager:
    """Manages the generation and display of heatmaps."""

    def __init__(self, result, output_directory="heatmaps"):
        self.result = result
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Attempt to auto-find log2 FC columns
        self.default_log2_fc_columns = [
            col for col in self.result.columns if 'log2' in col.lower()
        ]

        self.field_selector = None
        self._setup_widgets()

    def set_field_selector(self, field_selector):
        """Allow selecting which fields to include in hover text."""
        self.field_selector = field_selector

    def _setup_widgets(self):
        unified_layout = {"width": "450px"}
        label_style = {"description_width": "150px"}

        self.fc_mode_dropdown = Dropdown(
            options=["auto", "manual"],
            value="auto",
            description="Heatmap coloring method:",
            layout=unified_layout,
            style=label_style
        )

        self.user_min_input = FloatText(
            value=-3.0,
            description="Min value for manual coloring:",
            layout=unified_layout,
            style=label_style
        )
        self.user_max_input = FloatText(
            value=3.0,
            description="Max value for manual coloring:",
            layout=unified_layout,
            style=label_style
        )
        self.fc_min_max_inputs = VBox([self.fc_mode_dropdown, self.user_min_input, self.user_max_input])

        class_vals = (
            self.result['Class / family'].dropna().unique()
            if 'Class / family' in self.result.columns else []
        )
        self.class_selection = widgets.SelectMultiple(
            options=class_vals,
            description="Select Classes:",
            layout=unified_layout,
            style=label_style
        )

        # Label field
        string_columns = self.result.select_dtypes(include='object').columns
        self.label_field_dropdown = Dropdown(
            options=string_columns,
            value="Accession" if "Accession" in string_columns else None,
            description="Label Field:",
            layout=unified_layout,
            style=label_style
        )

        # Log2 FC columns (multiple selection)
        self.log2_fc_select = widgets.SelectMultiple(
            options=self.default_log2_fc_columns,
            value=self.default_log2_fc_columns if self.default_log2_fc_columns else (),
            description="Log2 FC Columns to use for plotting:",
            layout=unified_layout,
            style=label_style
        )

        if not self.default_log2_fc_columns:
            print("No default Log2 FC columns found. Please select from the dropdown.")

        self.generate_button = widgets.Button(description="Generate Heatmaps", button_style="success")
        self.generate_button.on_click(self._on_generate_click)

        self.output = Output()

    def display_interface(self):
        display(VBox([
            self.fc_min_max_inputs,
            self.log2_fc_select,
            self.class_selection,
            self.label_field_dropdown,
            self.generate_button,
            self.output
        ]))

    def _on_generate_click(self, b):
        with self.output:
            self.output.clear_output()
            try:
                fc_mode = self.fc_mode_dropdown.value
                user_defined_min = self.user_min_input.value
                user_defined_max = self.user_max_input.value
                selected_classes = list(self.class_selection.value)
                user_log2_fc_cols = list(self.log2_fc_select.value)

                selected_hover_fields = []
                if self.field_selector is not None:
                    selected_hover_fields = self.field_selector.get_selected_hover_fields()

                label_field = self.label_field_dropdown.value

                if not user_log2_fc_cols:
                    print("No log2 FC columns selected. Cannot generate heatmaps.")
                    return

                if not selected_classes:
                    print("Please select at least one class to generate heatmaps.")
                    return

                if not selected_hover_fields:
                    print("Please select hover fields before generating heatmaps.")
                    return

                print(f"Generating heatmaps with {fc_mode} coloring mode...")
                print(f"Selected classes: {selected_classes}")
                print(f"Selected hover fields: {selected_hover_fields}")
                print(f"Label field: {label_field}")
                print(f"Using these log2 FC columns: {user_log2_fc_cols}")

                for cls in selected_classes:
                    if 'Class / family' not in self.result.columns:
                        print(f"No 'Class / family' column found. Skipping class: {cls}")
                        continue

                    filtered_data = self.result.loc[self.result['Class / family'] == cls].copy()
                    log2_fc_columns = [c for c in user_log2_fc_cols if c in filtered_data.columns]
                    if not log2_fc_columns:
                        print(f"No overlap of chosen log2 FC columns for class: {cls}")
                        continue

                    if fc_mode == "auto":
                        max_abs_fc = filtered_data[log2_fc_columns].abs().max().max()
                        log2_fc_min = -max_abs_fc
                        log2_fc_max = max_abs_fc
                    else:
                        log2_fc_min = user_defined_min
                        log2_fc_max = user_defined_max

                    # Create a standard Figure (not FigureWidget)
                    fig = self._create_heatmap(
                        filtered_data,
                        log2_fc_columns,
                        selected_hover_fields,
                        cls,
                        log2_fc_min,
                        log2_fc_max,
                        label_field
                    )

                    # Use a standard Figure display (no FigureWidget)
                    save_button = widgets.Button(
                        description=f"Save Heatmap: {cls}",
                        button_style="success"
                    )
                    save_button.on_click(self.create_save_callback(fig, cls, log2_fc_columns[0]))

                    # Display the save button and the figure
                    display(VBox([save_button]))
                    display(fig)

            except Exception as e:
                print(f"Error generating heatmaps: {e}")

    def _create_heatmap(self, filtered_data, log2_fc_columns, hover_fields, cls,
                        log2_fc_min, log2_fc_max, label_field):
        # Example row-sorting logic
        sorted_data = filtered_data.copy()
        sorted_data['mean_fc'] = sorted_data[log2_fc_columns].mean(axis=1)
        sorted_data.sort_values('mean_fc', ascending=True, inplace=True)
        sorted_data.drop(columns=['mean_fc'], inplace=True)

        hover_texts = [
            [
                '<br>'.join(
                    [f"{field}: {sorted_data.iloc[row][field]}" for field in hover_fields] +
                    [f"{col}: {sorted_data.iloc[row][col]:.2f}"]
                )
                for col in log2_fc_columns
            ]
            for row in range(len(sorted_data))
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=sorted_data[log2_fc_columns].values,
                x=log2_fc_columns,
                y=sorted_data[label_field],
                text=hover_texts,
                hoverinfo="text",
                colorscale=['#8b0000', '#ffffff', '#404080'],
                zmin=log2_fc_min,
                zmax=log2_fc_max,
                colorbar=dict(
                    title="Log2 Fold Change",
                    tickvals=[log2_fc_min, 0, log2_fc_max],
                    ticktext=[f"{log2_fc_min:.2f}", "0", f"{log2_fc_max:.2f}"]
                ),
                hoverongaps=False,
                xgap=2,
                ygap=2
            )
        )

        # Retain your original layout
        fig.update_layout(
            title=f"<b>Heatmap for Class</b>: {cls}",
            xaxis=dict(
                title="Conditions",
                tickangle=45,
                titlefont=dict(size=12),
                range=[-10, len(log2_fc_columns) + 10],
                showgrid=False
            ),
            yaxis=dict(
                title=f"Proteins ({label_field})",
                titlefont=dict(size=12),
                tickfont=dict(size=10),
                showgrid=False,
                scaleanchor="x",
                scaleratio=1,
                automargin=True,
                anchor="x"
            ),
            template="plotly_white",
            width=900,
            height=max(600, 30 * len(sorted_data)),
            margin=dict(l=200, r=20, t=50, b=50)
        )
        return fig

    def create_save_callback(self, current_fig, class_name, log2_fc_column):
        def save_callback(_):
            try:
                sanitized_name = class_name.replace(' ', '_').replace('/', '_')
                sanitized_column = log2_fc_column.replace(' ', '_').replace('/', '_')
                base_filename = f"heatmap_{sanitized_name}_{sanitized_column}"
                html_path = os.path.join(self.output_directory, f"{base_filename}.html")
                svg_path = os.path.join(self.output_directory, f"{base_filename}.svg")

                current_fig.write_html(html_path)
                current_fig.write_image(svg_path, format="svg")

                print("Heatmap saved successfully!")
                print(f"HTML: {os.path.abspath(html_path)}")
                print(f"SVG: {os.path.abspath(svg_path)}")
            except Exception as e:
                print(f"Error saving heatmap: {e}")
        return save_callback



###############################################################################
# 6) Exports
###############################################################################
__all__ = [
    'DataLoader',
    'InteractiveFieldSelector',
    'VolcanoPlotManager',
    'HeatmapManager',
    'filter_columns_by_keywords'
]
