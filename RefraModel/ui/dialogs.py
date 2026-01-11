"""
Custom dialogs for the application
"""
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFormLayout,
    QLineEdit,
    QDialogButtonBox,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QCheckBox,
)
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt


class CustomDialog(QDialog):
    """A custom dialog for user interactions"""
    
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle("Information")
        self.layout = QVBoxLayout()
        
        self.label = QLabel(message)
        self.layout.addWidget(self.label)
        
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)
        
        self.setLayout(self.layout)


class StartParamsDialog(QDialog):
    """Dialog to capture startup parameters.

    When threshold_only is True, shows only the threshold field.
    Otherwise shows X/Y min/max, threshold, number of properties, and property details.
    """

    def __init__(self, threshold_only=False, defaults=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initial Parameters")
        self.threshold_only = threshold_only
        defaults = defaults or {}

        self.form = QFormLayout()

        dv = QDoubleValidator()

        self.xmin_edit = None
        self.xmax_edit = None
        self.ymin_edit = None
        self.ymax_edit = None
        self.nprops_edit = None
        self.prop_name_edits = []
        self.prop_value_edits = []
        self.save_file_edit = None

        if not self.threshold_only:
            self.xmin_edit = QLineEdit(str(defaults.get("xmin", 0.0)))
            self.xmin_edit.setValidator(dv)
            self.form.addRow("X_min [m]", self.xmin_edit)

            self.xmax_edit = QLineEdit(str(defaults.get("xmax", 100.0)))
            self.xmax_edit.setValidator(dv)
            self.form.addRow("X_max [m]", self.xmax_edit)

            self.ymin_edit = QLineEdit(str(defaults.get("ymin", -30.0)))
            self.ymin_edit.setValidator(dv)
            self.form.addRow("Y_min [m]", self.ymin_edit)

            self.ymax_edit = QLineEdit(str(defaults.get("ymax", 0.0)))
            self.ymax_edit.setValidator(dv)
            self.form.addRow("Y_max [m]", self.ymax_edit)

            # Number of properties
            self.nprops_edit = QLineEdit(str(defaults.get("nprops", 1)))
            from PyQt5.QtGui import QIntValidator
            self.nprops_edit.setValidator(QIntValidator(1, 10))
            self.nprops_edit.textChanged.connect(self._update_property_fields)
            self.form.addRow("Number of properties", self.nprops_edit)

            # Initial property fields
            self._update_property_fields()

        self.threshold_edit = QLineEdit(str(defaults.get("threshold", 1.0)))
        self.threshold_edit.setValidator(dv)
        self.form.addRow("Threshold for\npoint equality [%]", self.threshold_edit)

        # Always show save-to file field
        self.save_file_edit = QLineEdit(str(defaults.get("save_file", "model.txt")))
        self.form.addRow("Save model to file", self.save_file_edit)

        layout = QVBoxLayout()
        layout.addLayout(self.form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def _update_property_fields(self):
        """Update property name/value fields based on number of properties."""
        if self.threshold_only or self.nprops_edit is None:
            return
        
        try:
            nprops = int(self.nprops_edit.text())
        except ValueError:
            nprops = 1
        
        # Remove existing property fields by searching for property rows
        rows_to_remove = []
        for row_idx in range(self.form.rowCount()):
            label_item = self.form.itemAt(row_idx, QFormLayout.LabelRole)
            if label_item and label_item.widget():
                label_text = label_item.widget().text()
                if label_text.startswith("Property ") and label_text[9:].split()[0].isdigit():
                    rows_to_remove.append(row_idx)
        
        # Remove in reverse order to maintain indices
        for row_idx in reversed(rows_to_remove):
            self.form.removeRow(row_idx)
        
        self.prop_name_edits = []
        self.prop_value_edits = []
        
        # Find threshold row index
        threshold_row = -1
        for row_idx in range(self.form.rowCount()):
            label_item = self.form.itemAt(row_idx, QFormLayout.LabelRole)
            if label_item and label_item.widget() and "Threshold" in label_item.widget().text():
                threshold_row = row_idx
                break
        
        # Add new property fields (before threshold)
        dv = QDoubleValidator()
        insert_at = threshold_row if threshold_row >= 0 else self.form.rowCount()
        for i in range(nprops):
            name_edit = QLineEdit(f"Property_{i+1}")
            value_edit = QLineEdit("1000.0")
            value_edit.setValidator(dv)
            
            # Create a container widget for name and value side-by-side
            container = QWidget()
            hlayout = QHBoxLayout(container)
            hlayout.setContentsMargins(0, 0, 0, 0)
            hlayout.addWidget(QLabel("Name:"))
            hlayout.addWidget(name_edit)
            hlayout.addWidget(QLabel("Value:"))
            hlayout.addWidget(value_edit)
            
            self.form.insertRow(insert_at + i, f"Property {i+1}", container)
            self.prop_name_edits.append(name_edit)
            self.prop_value_edits.append(value_edit)

    def values(self):
        """Return parsed values as dict or None if invalid."""
        try:
            vals = {
                "threshold": float(self.threshold_edit.text()),
                "save_file": self.save_file_edit.text().strip() or "model.txt",
            }
            if not self.threshold_only:
                vals.update({
                    "xmin": float(self.xmin_edit.text()),
                    "xmax": float(self.xmax_edit.text()),
                    "ymin": float(self.ymin_edit.text()),
                    "ymax": float(self.ymax_edit.text()),
                })
                # Collect property names and values
                prop_names = []
                prop_values = []
                for name_edit, value_edit in zip(self.prop_name_edits, self.prop_value_edits):
                    prop_names.append(name_edit.text().strip() or "Property")
                    prop_values.append(float(value_edit.text()))
                vals["prop_names"] = prop_names
                vals["prop_values"] = prop_values
            return vals
        except Exception:
            return None


class PropertyEditorDialog(QDialog):
    """Dialog to edit a body's name and property values.

    Shows the body index in the title, the current name, and one input per property value.
    Property names are shown as labels; values are editable and validated as floats.
    """

    def __init__(self, body_index, body_name, prop_names, prop_values, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Properties — Body {body_index}")
        try:
            self.setMinimumWidth(360)
        except Exception:
            pass

        self.name_edit = QLineEdit(body_name or "")

        self.value_edits = []
        dv = QDoubleValidator()

        form = QFormLayout()
        # Show body index inside the dialog so it's always visible
        from PyQt5.QtWidgets import QLabel as _QLabel
        form.addRow("Body", _QLabel(str(body_index)))
        form.addRow("Name", self.name_edit)

        # Ensure parallel lengths
        prop_names = list(prop_names or [])
        prop_values = list(prop_values or [])
        n = max(len(prop_names), len(prop_values))
        if len(prop_names) < n:
            prop_names += [f"Property_{i+1}" for i in range(len(prop_names), n)]
        if len(prop_values) < n:
            prop_values += [0.0 for _ in range(len(prop_values), n)]

        for name, val in zip(prop_names, prop_values):
            le = QLineEdit(str(val))
            le.setValidator(dv)
            form.addRow(name, le)
            self.value_edits.append(le)

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self._prop_names = prop_names

    def values(self):
        """Return updated name and property values as (name, values)."""
        try:
            name = self.name_edit.text().strip()
            values = [float(le.text()) for le in self.value_edits]
            return name, values
        except Exception:
            return None, None


class SelectPickfDialog(QDialog):
    """Dialog to select receiver/shot decimation and starts.

    Fields:
    - Every n_th receiver (xxx available)
    - Starting with (natural counting)
    - Every n_th shot (yyy available)
    - Starting with (natural counting)
    """

    def __init__(self, n_receivers: int, n_shots: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select picks")

        form = QFormLayout()

        iv_pos = QIntValidator(1, max(1, n_receivers))
        iv_shot = QIntValidator(1, max(1, n_shots))
        iv_any = QIntValidator(1, 10**6)

        self.dg_edit = QLineEdit("1")
        self.dg_edit.setValidator(iv_any)
        form.addRow(f"Every n_th receiver\n{n_receivers} available", self.dg_edit)

        self.g0_edit = QLineEdit("1")
        self.g0_edit.setValidator(iv_pos)
        form.addRow("Starting with (natural counting)", self.g0_edit)

        self.ds_edit = QLineEdit("1")
        self.ds_edit.setValidator(iv_any)
        form.addRow(f"Every n_th shot\n{n_shots} available", self.ds_edit)

        self.s0_edit = QLineEdit("1")
        self.s0_edit.setValidator(iv_shot)
        form.addRow("Starting with (natural counting)", self.s0_edit)

        layout = QVBoxLayout()
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def values(self):
        try:
            dg = int(self.dg_edit.text())
            g0 = int(self.g0_edit.text())
            ds = int(self.ds_edit.text())
            s0 = int(self.s0_edit.text())
            return g0, dg, s0, ds
        except Exception:
            return None


class InversionParametersDialog(QDialog):
    """Dialog to set inversion parameters.
    
    Collects:
    - Starting model choice (radio buttons)
    - Gradient model parameters (conditional)
    - Inversion control parameters
    """
    
    # Class variable to store defaults across instances
    _defaults = {
        'use_actual_model': True,
        'v_top': 200.0,
        'v_bottom': 4000.0,
        'abort_chi2': True,
        'max_delta_phi': 2.0,
        'max_iterations': 10,
        'min_velocity': 200.0,
        'max_velocity': 6000.0,
        'initial_lambda': 200.0,
        'lambda_reduction': 0.7,
        'z_smoothing': 0.2,
        'velocity_limit_percent': 50.0,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inversion Parameters")
        
        layout = QVBoxLayout()
        form = QFormLayout()
        
        # Starting model label
        form.addRow(QLabel("<b>Starting model:</b>"))
        
        # Radio buttons for starting model
        self.rb_actual = QRadioButton("Use actual model")
        self.rb_gradient = QRadioButton("Gradient model")
        
        self.rb_actual.setChecked(self._defaults['use_actual_model'])
        self.rb_gradient.setChecked(not self._defaults['use_actual_model'])
        
        self.rb_actual.toggled.connect(self._update_gradient_fields)
        
        form.addRow(self.rb_actual)
        form.addRow(self.rb_gradient)
        
        # Gradient model parameters (conditional)
        self.v_top_edit = QLineEdit(str(self._defaults['v_top']))
        self.v_top_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.v_top_label = QLabel("v_top [m/s]")
        
        self.v_bottom_edit = QLineEdit(str(self._defaults['v_bottom']))
        self.v_bottom_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        self.v_bottom_label = QLabel("v_bottom [m/s]")
        
        self.v_top_row = form.rowCount()
        form.addRow(self.v_top_label, self.v_top_edit)
        self.v_bottom_row = form.rowCount()
        form.addRow(self.v_bottom_label, self.v_bottom_edit)
        
        # Chi-squared abort checkbox
        self.abort_chi2_check = QCheckBox("Abort when chi² <= 1")
        self.abort_chi2_check.setChecked(self._defaults['abort_chi2'])
        form.addRow(self.abort_chi2_check)
        
        # Convergence control - maximum delta phi
        self.max_delta_phi_edit = QLineEdit(str(self._defaults['max_delta_phi']))
        self.max_delta_phi_edit.setValidator(QDoubleValidator(0.0, 100.0, 2))
        form.addRow("Maximum delta_phi [%]", self.max_delta_phi_edit)
        
        # Maximum iterations
        self.max_iter_edit = QLineEdit(str(self._defaults['max_iterations']))
        self.max_iter_edit.setValidator(QIntValidator(0, 1000))
        form.addRow("Maximum number of iterations (0=automatic)", self.max_iter_edit)
        
        # Separator for stopping conditions
        form.addRow(QLabel(""))
        form.addRow(QLabel("<b>Model constraints:</b>"))
        
        # Velocity constraints
        self.min_vel_edit = QLineEdit(str(self._defaults['min_velocity']))
        self.min_vel_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        form.addRow("Minimum allowed velocity [m/s]", self.min_vel_edit)
        
        self.max_vel_edit = QLineEdit(str(self._defaults['max_velocity']))
        self.max_vel_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        form.addRow("Maximum allowed velocity [m/s]", self.max_vel_edit)
        
        # Smoothing parameters
        self.lambda_edit = QLineEdit(str(self._defaults['initial_lambda']))
        self.lambda_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        form.addRow("Initial smoothing parameter", self.lambda_edit)
        
        self.lambda_reduce_edit = QLineEdit(str(self._defaults['lambda_reduction']))
        self.lambda_reduce_edit.setValidator(QDoubleValidator(0.0, 1.0, 3))
        form.addRow("Smoothing reduction factor per iteration", self.lambda_reduce_edit)
        
        self.z_smooth_edit = QLineEdit(str(self._defaults['z_smoothing']))
        self.z_smooth_edit.setValidator(QDoubleValidator(0.0, 1.0, 3))
        form.addRow("Relative smoothing in z direction (0 to 1)", self.z_smooth_edit)
        
        # Velocity variation limit (per body)
        self.velocity_limit_edit = QLineEdit(str(self._defaults['velocity_limit_percent']))
        self.velocity_limit_edit.setValidator(QDoubleValidator(0.0, 100.0, 1))
        self.velocity_limit_edit.setToolTip("Each body is constrained to ± this percentage of its starting velocity.\n0% means use global min/max velocity limits instead.")
        form.addRow("Limit velocity variations per body to ± x%\n(0=use global limits)", self.velocity_limit_edit)
        
        # Anisotropic smoothing
        form.addRow(QLabel(""))
        form.addRow(QLabel("<b>Anisotropic smoothing:</b>"))
        
        self.aniso_check = QCheckBox("Use anisotropic smoothing")
        self.aniso_check.setChecked(self._defaults.get('anisotropic', False))
        self.aniso_check.toggled.connect(self._update_aniso_fields)
        form.addRow(self.aniso_check)
        
        self.horiz_corr_edit = QLineEdit(str(self._defaults.get('horizontal_correlation', 10.0)))
        self.horiz_corr_edit.setValidator(QDoubleValidator(0.1, 1000.0, 1))
        self.horiz_corr_label = QLabel("Horizontal correlation length [m]")
        form.addRow(self.horiz_corr_label, self.horiz_corr_edit)
        
        self.vert_corr_edit = QLineEdit(str(self._defaults.get('vertical_correlation', 1.0)))
        self.vert_corr_edit.setValidator(QDoubleValidator(0.1, 1000.0, 1))
        self.vert_corr_label = QLabel("Vertical correlation length [m]")
        form.addRow(self.vert_corr_label, self.vert_corr_edit)
        
        layout.addLayout(form)
        
        # Update visibility of gradient and anisotropic fields
        self._update_gradient_fields()
        self._update_aniso_fields()
        
        # Checkbox to run inversion immediately
        self.do_inversion_check = QCheckBox("Do inversion")
        self.do_inversion_check.setChecked(True)
        layout.addWidget(self.do_inversion_check)
        
        # Checkbox to modify regularization in specific bodies
        self.modify_regularization_check = QCheckBox("Modify regularization\nin specific bodies")
        self.modify_regularization_check.setChecked(False)
        layout.addWidget(self.modify_regularization_check)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def _update_gradient_fields(self):
        """Show/hide gradient model parameter fields based on radio button selection."""
        show_gradient = self.rb_gradient.isChecked()
        self.v_top_label.setVisible(show_gradient)
        self.v_top_edit.setVisible(show_gradient)
        self.v_bottom_label.setVisible(show_gradient)
        self.v_bottom_edit.setVisible(show_gradient)
    
    def _update_aniso_fields(self):
        """Show/hide anisotropic correlation length fields based on checkbox."""
        show_aniso = self.aniso_check.isChecked()
        self.horiz_corr_label.setVisible(show_aniso)
        self.horiz_corr_edit.setVisible(show_aniso)
        self.vert_corr_label.setVisible(show_aniso)
        self.vert_corr_edit.setVisible(show_aniso)
    
    def accept(self):
        """Update defaults when user clicks OK."""
        params = self.get_parameters()
        InversionParametersDialog._defaults.update(params)
        super().accept()
    
    def get_parameters(self):
        """Return the selected parameters as a dictionary."""
        params = {
            'use_actual_model': self.rb_actual.isChecked(),
            'v_top': float(self.v_top_edit.text()),
            'v_bottom': float(self.v_bottom_edit.text()),
            'abort_chi2': self.abort_chi2_check.isChecked(),
            'max_delta_phi': float(self.max_delta_phi_edit.text()),
            'max_iterations': int(self.max_iter_edit.text()),
            'min_velocity': float(self.min_vel_edit.text()),
            'max_velocity': float(self.max_vel_edit.text()),
            'initial_lambda': float(self.lambda_edit.text()),
            'lambda_reduction': float(self.lambda_reduce_edit.text()),
            'z_smoothing': float(self.z_smooth_edit.text()),
            'velocity_limit_percent': float(self.velocity_limit_edit.text()),
            'anisotropic': self.aniso_check.isChecked(),
            'horizontal_correlation': float(self.horiz_corr_edit.text()),
            'vertical_correlation': float(self.vert_corr_edit.text()),
            'do_inversion': self.do_inversion_check.isChecked(),
            'modify_regularization': self.modify_regularization_check.isChecked(),
        }
        return params


class GeometryDialog(QDialog):
    """Dialog to create acquisition geometry when no pick file exists.
    
    Collects receiver and shot point geometry parameters.
    """

    def __init__(self, x_min, x_max, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Acquisition Geometry")
        self.x_min = x_min
        self.x_max = x_max
        
        layout = QVBoxLayout()
        form = QFormLayout()
        
        # Receiver parameters
        form.addRow(QLabel("<b>Receivers:</b>"))
        
        self.rx_start_edit = QLineEdit("0.0")
        self.rx_start_edit.setValidator(QDoubleValidator())
        form.addRow("X-coordinate of first receiver [m]", self.rx_start_edit)
        
        self.rx_spacing_edit = QLineEdit("1.0")
        self.rx_spacing_edit.setValidator(QDoubleValidator(0.01, 1000.0, 2))
        self.rx_spacing_edit.textChanged.connect(self._update_max_receivers)
        form.addRow("Distance between receivers [m]", self.rx_spacing_edit)
        
        # Calculate initial max receivers
        max_rx = self._calculate_max_receivers()
        self.rx_num_edit = QLineEdit(str(max_rx))
        self.rx_num_edit.setValidator(QIntValidator(1, max_rx))
        self.rx_num_label = QLabel(f"Number of receivers (max = {max_rx})")
        form.addRow(self.rx_num_label, self.rx_num_edit)
        
        # Shot point parameters
        form.addRow(QLabel(""))  # Spacer
        form.addRow(QLabel("<b>Shot points:</b>"))
        
        self.shot_start_edit = QLineEdit("0.0")
        self.shot_start_edit.setValidator(QDoubleValidator())
        form.addRow("X-coordinate of first shot point [m]", self.shot_start_edit)
        
        self.shot_spacing_edit = QLineEdit("2.0")
        self.shot_spacing_edit.setValidator(QDoubleValidator(0.01, 1000.0, 2))
        self.shot_spacing_edit.textChanged.connect(self._update_max_shots)
        form.addRow("Distance between shot points [m]", self.shot_spacing_edit)
        
        # Calculate initial max shots
        max_shots = self._calculate_max_shots()
        self.shot_num_edit = QLineEdit(str(max_shots))
        self.shot_num_edit.setValidator(QIntValidator(1, max_shots))
        self.shot_num_label = QLabel(f"Number of shot points (max = {max_shots})")
        form.addRow(self.shot_num_label, self.shot_num_edit)
        
        layout.addLayout(form)
        
        # Checkbox to save calculated times to file
        self.save_to_file_check = QCheckBox("Store calculated travel times\nto file picks.sgt")
        self.save_to_file_check.setChecked(True)  # Checked by default
        layout.addWidget(self.save_to_file_check)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
    
    def _calculate_max_receivers(self):
        """Calculate maximum number of receivers based on spacing."""
        try:
            spacing = float(self.rx_spacing_edit.text())
            if spacing > 0:
                return int((self.x_max - self.x_min) / spacing) + 1
        except (ValueError, ZeroDivisionError):
            pass
        return 100  # Fallback
    
    def _calculate_max_shots(self):
        """Calculate maximum number of shot points based on spacing."""
        try:
            spacing = float(self.shot_spacing_edit.text())
            if spacing > 0:
                return int((self.x_max - self.x_min) / spacing) + 1
        except (ValueError, ZeroDivisionError):
            pass
        return 50  # Fallback
    
    def _update_max_receivers(self):
        """Update max receivers label and validator when spacing changes."""
        max_rx = self._calculate_max_receivers()
        self.rx_num_label.setText(f"Number of receivers (max = {max_rx})")
        
        # Always update the value to the new max
        self.rx_num_edit.setText(str(max_rx))
        
        # Update validator for future manual edits
        self.rx_num_edit.setValidator(QIntValidator(1, max_rx))
    
    def _update_max_shots(self):
        """Update max shots label and validator when spacing changes."""
        max_shots = self._calculate_max_shots()
        self.shot_num_label.setText(f"Number of shot points (max = {max_shots})")
        
        # Always update the value to the new max
        self.shot_num_edit.setText(str(max_shots))
        
        # Update validator for future manual edits
        self.shot_num_edit.setValidator(QIntValidator(1, max_shots))
    
    def get_parameters(self):
        """Return the geometry parameters as a dictionary."""
        params = {
            'rx_start': float(self.rx_start_edit.text()),
            'rx_spacing': float(self.rx_spacing_edit.text()),
            'rx_count': int(self.rx_num_edit.text()),
            'shot_start': float(self.shot_start_edit.text()),
            'shot_spacing': float(self.shot_spacing_edit.text()),
            'shot_count': int(self.shot_num_edit.text()),
            'save_to_file': self.save_to_file_check.isChecked(),
        }
        return params


class BodyRegularizationDialog(QDialog):
    """Dialog to set body-specific regularization parameters."""
    
    def __init__(self, body_index, velocity, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Regularization for Body {body_index}")
        self.body_index = body_index
        self.velocity = velocity
        
        layout = QVBoxLayout()
        
        # Info label
        info_label = QLabel(f"<b>Body {body_index}</b> - Current velocity: {velocity:.1f} m/s")
        layout.addWidget(info_label)
        
        # Regularization type selection
        reg_group = QGroupBox("Regularization Type")
        reg_layout = QVBoxLayout()
        
        self.free_radio = QRadioButton("Free inversion")
        self.free_radio.setChecked(True)
        self.free_radio.toggled.connect(self._on_type_changed)
        reg_layout.addWidget(self.free_radio)
        
        self.single_radio = QRadioButton("Single velocity (strong smoothing)")
        self.single_radio.toggled.connect(self._on_type_changed)
        reg_layout.addWidget(self.single_radio)
        
        self.range_radio = QRadioButton("Velocity range")
        self.range_radio.toggled.connect(self._on_type_changed)
        reg_layout.addWidget(self.range_radio)
        
        self.fixed_radio = QRadioButton("Fixed velocity")
        self.fixed_radio.toggled.connect(self._on_type_changed)
        reg_layout.addWidget(self.fixed_radio)
        
        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)
        
        # Parameters group (shown/hidden based on selection)
        self.params_group = QGroupBox("Parameters")
        params_layout = QFormLayout()
        
        # Single velocity field (for single and fixed modes)
        self.velocity_edit = QLineEdit(str(velocity))
        self.velocity_edit.setValidator(QDoubleValidator(100.0, 10000.0, 1))
        params_layout.addRow("Velocity [m/s]:", self.velocity_edit)
        
        # Velocity range fields (for range mode)
        self.vmin_edit = QLineEdit(str(max(100, velocity * 0.8)))
        self.vmin_edit.setValidator(QDoubleValidator(100.0, 10000.0, 1))
        params_layout.addRow("Minimum velocity [m/s]:", self.vmin_edit)
        
        self.vmax_edit = QLineEdit(str(velocity * 1.2))
        self.vmax_edit.setValidator(QDoubleValidator(100.0, 10000.0, 1))
        params_layout.addRow("Maximum velocity [m/s]:", self.vmax_edit)
        
        # Velocity variation limit (for free mode only)
        self.velocity_limit_edit = QLineEdit("50.0")
        self.velocity_limit_edit.setValidator(QDoubleValidator(0.0, 100.0, 1))
        self.velocity_limit_edit.setToolTip("Override the percentage from main dialog for this specific body.\n0% means use global min/max velocity limits.")
        params_layout.addRow("Limit velocity to ± x%\n(0=use global limits):", self.velocity_limit_edit)
        
        self.params_group.setLayout(params_layout)
        layout.addWidget(self.params_group)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
        # Initial UI state
        self._on_type_changed()
    
    def _on_type_changed(self):
        """Update UI based on selected regularization type."""
        if self.free_radio.isChecked():
            self.params_group.setVisible(True)
            self.velocity_edit.setEnabled(False)
            self.vmin_edit.setEnabled(False)
            self.vmax_edit.setEnabled(False)
            self.velocity_limit_edit.setEnabled(True)
            # Set default to 50% (matching main dialog default)
            if self.velocity_limit_edit.text() == "0.0":
                self.velocity_limit_edit.setText("50.0")
        elif self.single_radio.isChecked():
            self.params_group.setVisible(True)
            self.velocity_edit.setEnabled(True)
            self.vmin_edit.setEnabled(False)
            self.vmax_edit.setEnabled(False)
            self.velocity_limit_edit.setEnabled(False)
        elif self.range_radio.isChecked():
            self.params_group.setVisible(True)
            self.velocity_edit.setEnabled(False)
            self.vmin_edit.setEnabled(True)
            self.vmax_edit.setEnabled(True)
            # Disable velocity limit and set to 0 for range mode
            self.velocity_limit_edit.setText("0.0")
            self.velocity_limit_edit.setEnabled(False)
        elif self.fixed_radio.isChecked():
            self.params_group.setVisible(True)
            self.velocity_edit.setEnabled(True)
            self.vmin_edit.setEnabled(False)
            self.vmax_edit.setEnabled(False)
            self.velocity_limit_edit.setEnabled(False)
    
    def get_parameters(self):
        """Return the regularization parameters as a dictionary."""
        params = {}
        
        if self.free_radio.isChecked():
            params = {
                'type': 'free',
                'velocity_limit_percent': float(self.velocity_limit_edit.text())
            }
        elif self.single_radio.isChecked():
            params = {
                'type': 'single',
                'velocity': float(self.velocity_edit.text())
            }
        elif self.range_radio.isChecked():
            params = {
                'type': 'range',
                'vmin': float(self.vmin_edit.text()),
                'vmax': float(self.vmax_edit.text()),
                'velocity_limit_percent': float(self.velocity_limit_edit.text())
            }
        elif self.fixed_radio.isChecked():
            params = {
                'type': 'fixed',
                'velocity': float(self.velocity_edit.text())
            }
        
        return params


class ColorScaleDialog(QDialog):
    """Dialog to configure color scale for inverted model display."""
    
    def __init__(self, current_vmin=None, current_vmax=None, current_cmap='jet', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Scale Settings")
        
        layout = QVBoxLayout()
        form = QFormLayout()
        
        # Velocity range
        self.vmin_edit = QLineEdit(str(current_vmin) if current_vmin else "")
        self.vmin_edit.setValidator(QDoubleValidator(0.0, 100000.0, 1))
        self.vmin_edit.setPlaceholderText("Auto")
        form.addRow("Minimum velocity [m/s]:", self.vmin_edit)
        
        self.vmax_edit = QLineEdit(str(current_vmax) if current_vmax else "")
        self.vmax_edit.setValidator(QDoubleValidator(0.0, 100000.0, 1))
        self.vmax_edit.setPlaceholderText("Auto")
        form.addRow("Maximum velocity [m/s]:", self.vmax_edit)
        
        layout.addLayout(form)
        
        # Colormap selection
        cmap_group = QGroupBox("Color Map")
        cmap_layout = QVBoxLayout()
        
        self.cmap_group = QButtonGroup(self)
        
        self.jet_radio = QRadioButton("jet (rainbow)")
        self.cmap_group.addButton(self.jet_radio, 0)
        cmap_layout.addWidget(self.jet_radio)
        
        self.spectral_radio = QRadioButton("Spectral_r (blue-green-yellow-red)")
        self.cmap_group.addButton(self.spectral_radio, 1)
        cmap_layout.addWidget(self.spectral_radio)
        
        self.viridis_radio = QRadioButton("viridis (blue-green-yellow)")
        self.cmap_group.addButton(self.viridis_radio, 2)
        cmap_layout.addWidget(self.viridis_radio)
        
        self.seismic_radio = QRadioButton("seismic (blue-white-red)")
        self.cmap_group.addButton(self.seismic_radio, 3)
        cmap_layout.addWidget(self.seismic_radio)
        
        # Set current colormap
        if current_cmap == 'Spectral_r':
            self.spectral_radio.setChecked(True)
        elif current_cmap == 'viridis':
            self.viridis_radio.setChecked(True)
        elif current_cmap == 'seismic':
            self.seismic_radio.setChecked(True)
        else:
            self.jet_radio.setChecked(True)
        
        cmap_group.setLayout(cmap_layout)
        layout.addWidget(cmap_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        reset_button = QPushButton("Reset to Auto")
        reset_button.clicked.connect(self._reset_to_auto)
        button_layout.addWidget(reset_button)
        
        button_layout.addStretch()
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _reset_to_auto(self):
        """Reset velocity range to automatic."""
        self.vmin_edit.clear()
        self.vmax_edit.clear()
    
    def get_parameters(self):
        """Return the color scale parameters as a dictionary."""
        # Get colormap
        if self.spectral_radio.isChecked():
            cmap = 'Spectral_r'
        elif self.viridis_radio.isChecked():
            cmap = 'viridis'
        elif self.seismic_radio.isChecked():
            cmap = 'seismic'
        else:
            cmap = 'jet'
        
        # Get velocity range (None if empty)
        vmin = float(self.vmin_edit.text()) if self.vmin_edit.text() else None
        vmax = float(self.vmax_edit.text()) if self.vmax_edit.text() else None
        
        return {
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cmap
        }