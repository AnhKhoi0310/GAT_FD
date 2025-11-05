from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QDoubleSpinBox, QPushButton, QCheckBox,
    QTabWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSpinBox, QGroupBox,
    QFormLayout, QMainWindow, QFrame
)
from PyQt6.QtGui import QIntValidator 
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QFileDialog
from scipy.io import savemat
import os
import numpy as np
# from nilearn.glm.first_level import spm_hrf
from PyQt6.QtWidgets import QMessageBox
from library.spm_hrf import spm_hrf as my_spm_hrf
class DesignWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.createComponents()
        self.parameters_default()
        self.parameters_update_field()
    #  Button pushed function: DefaulSettingButton
    def parameters_default(self):
        self.gatd_plot = type('', (), {})()
        self.settings = {
        'tr': 1,
        'window_size': 20,
        'step_size': 1,
        # Design
        'design_condition_list': '',
        'design_duration_list': '',
        # Design Auto/Manual
        'stage_method': 1,  # 1: auto; 2: manual
        # Condition
        'activation_enable': 1,
        'activation_level': 0.8,
        'activation_percent': 80,
        'condition_enable': 1,
        'condition_percent': 80,
        'stage_condition_list': '',
        'stage_duration_list': ''
    }

    def parameters_update_field(self):
        # Sliding Window Setting
        self.tr_value.setText(str(self.settings['tr']))
        self.window_size.setText(str(self.settings['window_size']))
        self.step_size.setText(str(self.settings['step_size']))

        # Design
        self.condition_sequence.setText(self.settings['design_condition_list'])
        self.duration_sequence.setText(self.settings['design_duration_list'])

        # Condition - Automatic Tab
        self.cb_activation.setChecked(bool(self.settings['activation_enable']))
        self.activation_level.setText(str(self.settings['activation_level']))
        self.activation_coverage.setText(str(self.settings['activation_percent']))
        self.cb_condition.setChecked(bool(self.settings['condition_enable']))
        self.condition_coverage.setText(str(self.settings['condition_percent']))

        # Condition - Manual Tab
        self.stage_condition.setText(self.settings['stage_condition_list'])
        self.stage_duration.setText(self.settings['stage_duration_list'])

    def parameters_update_setting(self):
        # Sliding Window Setting
        try:
            self.settings['tr'] = int(self.tr_value.text())
        except ValueError:
            self.settings['tr'] = 1
        # self.settings['window_size'] = int(self.window_size.text())
        try:
            self.settings['window_size'] = int(self.window_size.text())
        except ValueError:
            self.settings['window_size'] = 1
            
        try:
            self.settings['step_size'] = int(self.step_size.text())
        except ValueError:
            self.settings['step_size'] = 1
        # self.settings['step_size'] = int(self.step_size.text())

        # Design
        self.settings['design_condition_list'] = self.condition_sequence.text()
        self.settings['design_duration_list'] = self.duration_sequence.text()

        # Determine which tab is selected
        current_tab = self.findChild(QTabWidget).currentIndex()
        current_tab_text = self.findChild(QTabWidget).tabText(current_tab)

        if current_tab_text == "Automatic":
            self.settings['stage_method'] = 1
            self.settings['activation_enable'] = int(self.cb_activation.isChecked())
            self.settings['activation_level'] = float(self.activation_level.text())
            self.settings['activation_percent'] = int(self.activation_coverage.text())
            self.settings['condition_enable'] = int(self.cb_condition.isChecked())
            self.settings['condition_percent'] = int(self.condition_coverage.text())
        elif current_tab_text == "Manual":
            self.settings['stage_method'] = 2
            self.settings['stage_condition_list'] = self.stage_condition.text()
            self.settings['stage_duration_list'] = self.stage_duration.text()

    # Button pushed function: DefaulSettingButton
    def gatd_default(self):
        self.parameters_default()
        self.parameters_update_field()

    # Value changed function: CheckBoxCondition
    def gatd_s_condition(self):
        self.settings['condition_enable'] = int(self.cb_condition.isChecked())

        if self.settings['condition_enable'] == 1:
            self.condition_coverage.setEnabled(True)
        else:
            self.condition_coverage.setEnabled(False)

    # Value changed function: CheckBoxActivation
    def gatd_s_activation(self):
        self.settings['activation_enable'] = int(self.cb_activation.isChecked())

        if self.settings['activation_enable'] == 1:
            self.activation_level.setEnabled(True)
            self.activation_coverage.setEnabled(True)
        else:
            self.activation_level.setEnabled(False)
            self.activation_coverage.setEnabled(False)  
    # Button pushed function: CalculateButton
    def gatd_update_frames(self):
        self.parameters_update_setting()
        try:
            # Parse the duration string into a list of numbers
            duration_str = self.settings['design_duration_list']
            design_duration_list_in_s = [float(val) for val in duration_str.strip().split()]
            
            dfnc_length = int(sum(design_duration_list_in_s) // self.settings['tr'])
            total_steps = dfnc_length - self.settings['window_size'] + 1
            self.total_steps.setText(str(total_steps))
        except Exception:
            self.total_steps.setText("0")

    def gatd_update_design(self):
        self.parameters_update_setting()
        try:
            dfnc_hrf, _ = my_spm_hrf(self.settings['tr'])
            # print(dfnc_hrf)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"HRF calculation failed: {e}")
            return

        # --- Parse design condition list ---
        if not self.settings['design_condition_list'].strip():
            QMessageBox.critical(self, "Error", "Please enter condition sequence")
            return

        try:
            run_design_condition_list = list(map(float, self.settings['design_condition_list'].split()))
        except:
            QMessageBox.critical(self, "Error", "Please enter the condition sequence with only one space between numbers")
            return

        # --- Parse duration list ---
        if not self.settings['design_duration_list'].strip():
            QMessageBox.critical(self, "Error", "Please enter duration sequence")
            return

        try:
            run_design_duration_list_in_s = list(map(float, self.settings['design_duration_list'].split()))
        except:
            QMessageBox.critical(self, "Error", "Please enter the duration sequence with only one space between numbers")
            return

        if len(run_design_condition_list) != len(run_design_duration_list_in_s):
            QMessageBox.critical(self, "Error", "The condition sequence and duration sequence do not match")
            return

        # --- If manual stage method ---
        if self.settings['stage_method'] == 2:
            if not self.settings['stage_condition_list'].strip():
                QMessageBox.critical(self, "Error", "Please enter condition sequence")
                return
            try:
                run_stage_condition_list = list(map(float, self.settings['stage_condition_list'].split()))
            except:
                QMessageBox.critical(self, "Error", "Please enter the condition sequence with only one space between numbers")
                return

            if not self.settings['stage_duration_list'].strip():
                QMessageBox.critical(self, "Error", "Please enter duration sequence")
                return
            try:
                run_stage_duration_list = list(map(float, self.settings['stage_duration_list'].split()))
            except:
                QMessageBox.critical(self, "Error", "Please enter the duration sequence with only one space between numbers")
                return

            if len(run_stage_condition_list) != len(run_stage_duration_list):
                QMessageBox.critical(self, "Error", "The condition sequence and duration sequence do not match")
                return

        # --- Build design matrix (MATLAB-style, 1-based) ---
        dfnc_length = int(np.floor(sum(run_design_duration_list_in_s) / self.settings['tr']))
        temp_duration_list_s = np.cumsum(run_design_duration_list_in_s)
        temp_duration_list = np.zeros(len(run_design_duration_list_in_s), dtype=int)
        for i in range(len(temp_duration_list) - 1):
            temp_duration_list[i] = int(round(temp_duration_list_s[i] / self.settings['tr']))
        temp_duration_list[-1] = int(np.floor(temp_duration_list_s[-1] / self.settings['tr']))
        run_design_duration_list = [temp_duration_list[0]] + list(np.diff(temp_duration_list))

        dfnc_design = np.zeros(dfnc_length)
        duration_list_cs = np.cumsum(run_design_duration_list).astype(int)
        # MATLAB: dfnc_design(1:duration_list_cs(1)) = run_design_condition_list(1)
        if len(duration_list_cs) > 0:
            dfnc_design[:duration_list_cs[0]] = run_design_condition_list[0]
        for idx in range(1, len(run_design_condition_list)):
            dfnc_design[duration_list_cs[idx-1]:duration_list_cs[idx]] = run_design_condition_list[idx]
        self.settings['dfnc_design'] = dfnc_design
        # --- Convolve with HRF (MATLAB-style, no normalization, keep block-like shape) ---
        # MATLAB's conv does not normalize, and the block design is 0/1, so the output should look like a block with rounded edges
        dfnc_response = np.convolve(dfnc_design, dfnc_hrf)[:dfnc_length]
        self.settings['dfnc_reponse'] = dfnc_response

        # --- Calculate Condition Matrix (MATLAB logic) ---
        if self.settings['stage_method'] == 1:
            dfnc_condi = (dfnc_response > self.settings['activation_level']).astype(float)
            dfnc_window_condi = np.ones(dfnc_length + 1 - self.settings['window_size'])

            if self.cb_activation.isChecked():
                for idx in range(len(dfnc_window_condi)):
                    if np.sum(dfnc_condi[idx:idx+self.settings['window_size']]) < (self.settings['window_size'] * self.settings['activation_percent'] / 100):
                        dfnc_window_condi[idx] = 0

            if self.cb_condition.isChecked():
                for idx in range(len(dfnc_window_condi)):
                    if np.sum(dfnc_design[idx:idx+self.settings['window_size']]) < (self.settings['window_size'] * self.settings['condition_percent'] / 100):
                        dfnc_window_condi[idx] = 0

        else:
            stage_length = int(sum(run_stage_duration_list))
            if stage_length == (dfnc_length + 1 - self.settings['window_size']):
                stage_design = np.zeros(stage_length)
                duration_list_cs = np.cumsum(run_stage_duration_list).astype(int)
                if len(duration_list_cs) > 0:
                    stage_design[:duration_list_cs[0]] = run_stage_condition_list[0]
                for idx in range(1, len(run_stage_condition_list)):
                    stage_design[duration_list_cs[idx-1]:duration_list_cs[idx]] = run_stage_condition_list[idx]
                dfnc_window_condi = stage_design
            else:
                QMessageBox.critical(self, "Error", "The specified stage does not match design and window size")
                return

        self.settings['dfnc_window_condi'] = dfnc_window_condi

        # MATLAB: dfnc_condi_with_window_n = conv(ones(1,window_size),dfnc_window_condi)
        dfnc_condi_with_window_n = np.convolve(np.ones(self.settings['window_size']), dfnc_window_condi, mode='full')[:dfnc_length]
        dfnc_condi_with_window = (dfnc_condi_with_window_n > 0).astype(float)
        # --- Plot (MATLAB-style, 1-based x-axis) ---
        ax = self.ax
        ax.clear()

        # Calculate design change points (MATLAB: find(diff(design)~=0))
        design_switch_point = np.where(np.diff(self.settings['dfnc_design']) != 0)[0]
        # MATLAB: design_time_point = [1, repelem(design_switch_point+0.5,2), dfnc_length]
        design_time_point = np.concatenate([[1], np.repeat(design_switch_point + 1.5, 2), [dfnc_length]])
        # MATLAB: design_value_point = [design(1), design(1), repelem(design(design_switch_point+1),2)]
        design_value_point = np.concatenate([
            [self.settings['dfnc_design'][0], self.settings['dfnc_design'][0]],
            np.repeat(self.settings['dfnc_design'][design_switch_point + 1], 2)
        ])
        if len(design_value_point) < len(design_time_point):
            design_value_point = np.append(design_value_point, self.settings['dfnc_design'][-1])
        else:
            design_value_point[-1] = self.settings['dfnc_design'][-1]

        # Fix mismatch in x and y lengths
        min_len = min(len(design_time_point), len(design_value_point))
        design_time_point = design_time_point[:min_len]
        design_value_point = design_value_point[:min_len]
        ax.set_xlim(1, dfnc_length)
        ax.set_ylim(-0.1, 1.1)

        self.gatd_plot.pholder_window = []
        self.gatd_plot.max_overlap_window = int(np.max(dfnc_condi_with_window_n))

        for i in range(1, self.gatd_plot.max_overlap_window + 1):
            temp_condi = (dfnc_condi_with_window_n == i).astype(float)
            window_switch_point = np.where(np.diff(temp_condi) != 0)[0]
            window_time_point = np.concatenate([[1], np.repeat(window_switch_point + 1.5, 2), [dfnc_length]])
            window_value_point = np.concatenate([
                [temp_condi[0], temp_condi[0]],
                np.repeat(temp_condi[window_switch_point + 1], 2)
            ])
            if len(window_value_point) < len(window_time_point):
                window_value_point = np.append(window_value_point, temp_condi[-1])
            else:
                window_value_point[-1] = temp_condi[-1]
            min_len = min(len(window_time_point), len(window_value_point))
            window_time_point = window_time_point[:min_len]
            window_value_point = window_value_point[:min_len]
            patch = ax.fill_between(
                window_time_point,
                window_value_point * 1.1,
                -0.1,
                color=(0.4, 1.0, 0.3),
                alpha=0.1 + i * 0.8 / self.gatd_plot.max_overlap_window
            )
            self.gatd_plot.pholder_window.append(patch)
        self.gatd_plot.pholder_hrf, = ax.plot(np.arange(1, dfnc_length + 1), dfnc_response, '--', color=(1, 0.32, 0.16), linewidth=2)
        self.gatd_plot.pholder_design, = ax.plot(design_time_point, design_value_point, 'k-', linewidth=1)
        self.canvas.draw()

        self.cb_task.setEnabled(True)
        self.cb_mask.setEnabled(True)
        self.cb_hrf.setEnabled(True)

    # Value changed function: TaskDesignCheckBox
    def gatd_p_task(self):
        value = self.cb_task.isChecked()
        if value:
            self.gatd_plot.pholder_design.set_visible(True)
        else:
            self.gatd_plot.pholder_design.set_visible(False)
        self.canvas.draw()
        
    # Value changed function: HemodynamicResponseCheckBox
    def gatd_p_hrf(self):
        value = self.cb_hrf.isChecked()
        if value:
            self.gatd_plot.pholder_hrf.set_visible(True)
        else:
            self.gatd_plot.pholder_hrf.set_visible(False)
        self.canvas.draw()
        
    def gatd_p_window(self):
        value = self.cb_mask.isChecked()

        if value:
            for i in range(self.gatd_plot.max_overlap_window):
                self.gatd_plot.pholder_window[i].set_visible(True)
        else:
            for i in range(self.gatd_plot.max_overlap_window):
                self.gatd_plot.pholder_window[i].set_visible(False)
        self.canvas.draw()
    # Button pushed function: SaveDesignMatrixButton
    def gatd_output_mat(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Select directory to save settings",
            "dynamic_condition.mat",
            "MAT files (*.mat)"
        )
        if not filename:
            return
        else:
            window_condition = self.settings
            if not filename.endswith('.mat'):
                filename += '.mat'
            savemat(filename, {'window_condition': window_condition}) 

    def createComponents(self):
        self.setWindowTitle("GAT-FD - Task Design v0.2a")
        self.setGeometry(100, 100, 600, 680)
        main_layout = QVBoxLayout()
        grid = QGridLayout()
        # Design Duration Sequence
        grid.addWidget(QLabel("Design Duration Sequence (s)"), 4, 0)
        self.duration_sequence = QLineEdit()
        grid.addWidget(self.duration_sequence, 4, 1)
        # Design Condition Sequence
        grid.addWidget(QLabel("Design Condition Sequence (0 for rest)"), 3, 0)
        self.condition_sequence = QLineEdit()
        grid.addWidget(self.condition_sequence, 3, 1)
        # TR
        grid.addWidget(QLabel("TR(s)"), 0, 0)
        self.tr_value = QLineEdit()
        self.tr_value.setValidator(QIntValidator())
        self.tr_value.setText("1")
        grid.addWidget(self.tr_value, 0, 1)
        # Window Size
        grid.addWidget(QLabel("Window Size (TR)"), 1, 0)
        self.window_size = QLineEdit()
        self.window_size.setValidator(QIntValidator())
        self.window_size.setText("20")
        grid.addWidget(self.window_size, 1, 1)
        # Step Size
        grid.addWidget(QLabel("Step Size (TR)"), 2, 0)
        self.step_size = QLineEdit()
        self.step_size.setValidator(QIntValidator())
        self.step_size.setText("1")
        grid.addWidget(self.step_size, 2, 1)
        # Task condition label
        grid.addWidget(QLabel("Task Condition Specification"), 5, 0)
        main_layout.addLayout(grid)
        # Tabs
        tabs = QTabWidget()
        auto_tab = QWidget()
        auto_layout = QGridLayout()
        # CheckBoxActivation
        self.cb_activation = QCheckBox()
        self.cb_activation.setChecked(True)
        self.cb_activation.clicked.connect(self.gatd_s_activation)
        auto_layout.addWidget(self.cb_activation, 0, 0)
        auto_layout.addWidget(QLabel("Estimated Activation Level Threshold (0-1)"), 0, 1)
        self.activation_level = QLineEdit() #Activation level
        self.activation_level.setValidator(QIntValidator())
        self.activation_level.setText("1")
        auto_layout.addWidget(self.activation_level, 0, 2)
        self.activation_coverage = QLineEdit() #Activation Coverage
        self.activation_coverage.setValidator(QIntValidator())
        self.activation_coverage.setText("80")
        auto_layout.addWidget(QLabel("Estimated Activation Coverage Percentage Threshold (%)"), 1, 1)
        auto_layout.addWidget(self.activation_coverage, 1, 2)
        # CheckBoxCondition
        self.cb_condition = QCheckBox()
        self.cb_condition.setChecked(True)
        self.cb_condition.clicked.connect(self.gatd_s_condition)
        auto_layout.addWidget(self.cb_condition, 2, 0)
        auto_layout.addWidget(QLabel("Condition Coverage Percentage Threshold (%)"), 2, 1)
        self.condition_coverage = QLineEdit()
        self.condition_coverage.setValidator(QIntValidator())
        self.condition_coverage.setText("80")
        auto_layout.addWidget(self.condition_coverage, 2, 2)

        auto_tab.setLayout(auto_layout)
        tabs.addTab(auto_tab, "Automatic")

        manual_tab = QWidget()
        manual_layout = QGridLayout()
        manual_layout.addWidget(QLabel("Stage Condition Sequence"), 0, 0)
        self.stage_condition = QLineEdit()
        manual_layout.addWidget(self.stage_condition, 0, 1)

        manual_layout.addWidget(QLabel("Stage Duration Sequence (TRs)"), 1, 0)
        self.stage_duration = QLineEdit()
        manual_layout.addWidget(self.stage_duration, 1, 1)

        manual_layout.addWidget(QLabel("With the current setting, the total number of frames is"), 2, 0)
        self.total_steps = QLineEdit()
        self.total_steps.setValidator(QIntValidator())
        self.total_steps.setText("0")
        self.total_steps.setReadOnly(True)
        manual_layout.addWidget(self.total_steps, 2, 1)

        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.clicked.connect(self.gatd_update_frames)
        manual_layout.addWidget(self.calculate_btn, 2, 2)
        manual_tab.setLayout(manual_layout)
        tabs.addTab(manual_tab, "Manual")

        main_layout.addWidget(tabs)
        # Buttons
        self.default_btn = QPushButton("Defaul Setting")
        self.default_btn.clicked.connect(self.gatd_default)
        self.update_btn = QPushButton("Update Design")
        self.update_btn.clicked.connect(self.gatd_update_design)
        self.save_btn = QPushButton("Save Design Matrix")
        self.save_btn.clicked.connect(self.gatd_output_mat)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.default_btn)
        btn_layout.addWidget(self.update_btn)
        btn_layout.addWidget(self.save_btn)

        main_layout.addLayout(btn_layout)
        # Axes placeholder (Matplotlib)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Design Block")
        self.ax.set_xlabel("Frame (TR)")
        self.ax.set_ylabel("Condition")
        main_layout.addWidget(self.canvas)
        # Plot Option Checkboxes
        plot_group = QGroupBox("Plot Option")
        plot_layout = QHBoxLayout()
        # TaskDesignCheckBox
        self.cb_task = QCheckBox("Task Design")
        self.cb_task.setChecked(True)
        self.cb_task.setEnabled(False)
        self.cb_task.clicked.connect(self.gatd_p_task)
        # HemodynamicResponseCheckBox
        self.cb_hrf = QCheckBox("Hemodynamic Response")
        self.cb_hrf.setChecked(True)
        self.cb_hrf.setEnabled(False)
        self.cb_hrf.clicked.connect(self.gatd_p_hrf)
        # TemporalMaskCheckBox
        self.cb_mask = QCheckBox("Temporal Mask")
        self.cb_mask.setChecked(True)
        self.cb_mask.setEnabled(False)
        self.cb_mask.clicked.connect(self.gatd_p_window)
        
        plot_layout.addWidget(self.cb_task)
        plot_layout.addWidget(self.cb_hrf)
        plot_layout.addWidget(self.cb_mask)
        plot_group.setLayout(plot_layout)

        main_layout.addWidget(plot_group)
        self.setLayout(main_layout)
