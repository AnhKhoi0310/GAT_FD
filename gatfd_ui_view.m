classdef gatfd_ui_view < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        GATFD_view_UIFigure            matlab.ui.Figure
        UIAxes_Network                 matlab.ui.control.UIAxes
        FrameSliderLabel               matlab.ui.control.Label
        FrameSlider                    matlab.ui.control.Slider
        LoadProcessedFileButton        matlab.ui.control.Button
        UIAxes_TaskDesign              matlab.ui.control.UIAxes
        UIAxes_Timeseries              matlab.ui.control.UIAxes
        LoadTemporalMaskButton         matlab.ui.control.Button
        LoadNetworkPropertyFileButton  matlab.ui.control.Button
        ThresholdDropDownLabel         matlab.ui.control.Label
        ThresholdDropDown              matlab.ui.control.DropDown
        MeasureDropDownLabel           matlab.ui.control.Label
        MeasureDropDown                matlab.ui.control.DropDown
        SubjectDropDownLabel           matlab.ui.control.Label
        SubjectDropDown                matlab.ui.control.DropDown
        UpdateButton                   matlab.ui.control.Button
        FrameSliderLabelCursor         matlab.ui.control.Label
    end

    
    properties (Access = private)
        gatv_setting % Description
        gatv_plot
        gatv_data
    end
    
    methods (Access = private)
        
        function results = update_display(app)
            try
                %imshow(squeeze(app.gatv_setting.networks(app.gatv_setting.frame_cur,:,:)),'Parent',app.UIAxes_Network);
                imagesc(app.UIAxes_Network,squeeze(app.gatv_setting.networks(app.gatv_setting.frame_cur,:,:)));
                %colormap(app.UIAxes_Network,jet(256));
                axis(app.UIAxes_Network,'square');
            end
            try
                delete(app.gatv_plot.pholder_window)
                w_start=app.gatv_setting.frame_cur;
                w_end=app.gatv_setting.frame_cur+app.gatv_setting.condition.window_size;
                app.gatv_plot.pholder_window=patch(app.UIAxes_TaskDesign,[w_start,w_end,w_end,w_start],[0 0 3 3],[1 0.96 0.4]);
                app.gatv_plot.pholder_window.FaceAlpha=0.4;
                app.gatv_plot.pholder_window.FaceColor=[1 0.96 0.4];
            end
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            movegui(app.GATFD_view_UIFigure,'southeast');
            
            app.gatv_setting.frame_cur=1;
            app.gatv_setting.frame_limit=[1,1];
            app.gatv_setting.iscondition=0;
            
        end

        % Button pushed function: LoadProcessedFileButton
        function gatv_load_file(app, event)
            [fnc_temp_file,fnc_temp_path] = uigetfile('*.mat','Select Conditions');
            if isequal(fnc_temp_file,0)
                return
            else
                load([fnc_temp_path,fnc_temp_file],'subj_data');
                app.gatv_setting.networks=subj_data.d_corr;
                app.gatv_setting.frame_limit=[1,length(subj_data.d_corr)];
                app.FrameSlider.Limits=[0,length(subj_data.d_corr)];
            end
        end

        % Value changed function: FrameSlider
        function gatv_update_network(app, event)
            app.gatv_setting.frame_cur = round(app.FrameSlider.Value);
            app.FrameSliderLabelCursor.Text = num2str(app.gatv_setting.frame_cur);
            app.update_display();
        end

        % Key press function: GATFD_view_UIFigure
        function GATFD_view_UIFigureKeyPress(app, event)
            key = event.Key;
            switch key
                case 'rightarrow'
                    if app.gatv_setting.frame_cur<app.gatv_setting.frame_limit(2)
                        app.gatv_setting.frame_cur=app.gatv_setting.frame_cur+1;
                    end
                case 'leftarrow'
                    if app.gatv_setting.frame_cur>app.gatv_setting.frame_limit(1)
                        app.gatv_setting.frame_cur=app.gatv_setting.frame_cur-1;
                    end
                otherwise
            end
            app.FrameSlider.Value=app.gatv_setting.frame_cur;
            app.FrameSliderLabelCursor.Text = num2str(app.gatv_setting.frame_cur);
            app.update_display();
        end

        % Button pushed function: LoadTemporalMaskButton
        function gatv_load_design(app, event)
            [fnc_temp_file,gat_temp_path] = uigetfile('*.mat;*.txt','Select Conditions');
            if isequal(fnc_temp_file,0)
                app.gatv_setting.iscondition=0;
                return
            end
            
            file_fullpath=fullfile(gat_temp_path,fnc_temp_file);
            temp_file=load(file_fullpath,'window_condition');
            app.gatv_setting.condition=temp_file.window_condition;
            app.gatv_setting.iscondition=1;
            
            run_design_duration_list_in_s=str2num(app.gatv_setting.condition.design_duration_list);
            dfnc_length=floor(sum(run_design_duration_list_in_s)/app.gatv_setting.condition.tr);
            %dfnc_length=sum(app.gatv_setting.condition.duration_list);
            dfnc_reponse_max=max(app.gatv_setting.condition.dfnc_reponse);
            app.gatv_plot.pholder_hrf=plot(app.UIAxes_TaskDesign,1:dfnc_length,app.gatv_setting.condition.dfnc_reponse,'--r','LineWidth',2);
            app.gatv_plot.pholder_hrf.Color=[1 0.32 0.16];
            axis(app.UIAxes_TaskDesign,[0,dfnc_length,0,dfnc_reponse_max+0.1]);
            hold(app.UIAxes_TaskDesign,'on');
            app.gatv_plot.pholder_design=plot(app.UIAxes_TaskDesign,1:dfnc_length,app.gatv_setting.condition.dfnc_design,'k','LineWidth',1);            
            hold(app.UIAxes_TaskDesign,'off');
            app.FrameSlider.Limits=[0,dfnc_length];
            w_start=app.gatv_setting.frame_cur;
            w_end=app.gatv_setting.frame_cur+app.gatv_setting.condition.window_size;
            app.gatv_plot.pholder_window=patch(app.UIAxes_TaskDesign,[w_start,w_end,w_end,w_start],[0 0 3 3],[1 0.96 0.4]);
            app.gatv_plot.pholder_window.FaceAlpha=0.2;
            app.gatv_plot.pholder_window.FaceColor=[1 0.96 0.4];
            app.gatv_data.data_length=length(app.gatv_setting.condition.dfnc_reponse);
            app.gatv_data.data_frame_length=length(app.gatv_setting.condition.dfnc_window_condi);
            app.gatv_data.data_length_diff=floor(app.gatv_data.data_length-app.gatv_data.data_frame_length)/2;
            
        end

        % Button pushed function: LoadNetworkPropertyFileButton
        function gatv_load_measure(app, event)
            [fnc_temp_file,fnc_temp_path] = uigetfile('*.mat','Select Conditions');
            if isequal(fnc_temp_file,0)
                return
            end
            file_fullpath=fullfile(fnc_temp_path,fnc_temp_file);
            try
                temp_data=load(file_fullpath,'dnet_data_data_mat_global');
                temp_data_thres_list=load(file_fullpath,'dnet_data_threshold_list');
                temp_data_file_list=load(file_fullpath,'dnet_data_files');
                temp_data_measures=load(file_fullpath,'dnet_data_measures_glob');
                temp_data_size=size(temp_data.dnet_data_data_mat_global,1,2,3,4);
                app.gatv_data.data=zeros(temp_data_size(1),temp_data_size(2),temp_data_size(3)+1,temp_data_size(4)+1);
                app.gatv_data.data(:,:,2:end,2:end)=temp_data.dnet_data_data_mat_global;
                app.gatv_data.data(:,:,1,2:end)=mean(temp_data.dnet_data_data_mat_global,3);
                app.gatv_data.data(:,:,:,1)=mean(app.gatv_data.data(:,:,:,2:end),4);
                app.ThresholdDropDown.Items = [{'Mean'},cellfun(@num2str,num2cell(temp_data_thres_list.dnet_data_threshold_list),'UniformOutput',false)];
                app.ThresholdDropDown.ItemsData = 1:length(app.ThresholdDropDown.Items);
                app.SubjectDropDown.Items = [{'GroupAverage'},temp_data_file_list.dnet_data_files];
                app.SubjectDropDown.ItemsData = 1:length(app.SubjectDropDown.Items);
                app.MeasureDropDown.Items = temp_data_measures.dnet_data_measures_glob;
                app.MeasureDropDown.ItemsData = 1:length(app.MeasureDropDown.Items);
            catch
                errordlg("No global measures to display","Error");
                return
            end

        end

        % Button pushed function: UpdateButton
        function gatv_update_timeseries(app, event)
            if isnumeric(app.MeasureDropDown.Value) && isnumeric(app.ThresholdDropDown.Value) && isnumeric(app.SubjectDropDown.Value)
                yyaxis(app.UIAxes_Timeseries,'left');
                temp_data=app.gatv_data.data(:,app.MeasureDropDown.Value,app.ThresholdDropDown.Value,app.SubjectDropDown.Value);
                if app.gatv_setting.iscondition
                    plot(app.UIAxes_Timeseries,app.gatv_data.data_length_diff:app.gatv_data.data_length_diff-1+app.gatv_data.data_frame_length,temp_data,'b','LineWidth',2);
                    ylim(app.UIAxes_Timeseries,'auto')
                    yyaxis(app.UIAxes_Timeseries,'right');
                    plot(app.UIAxes_Timeseries,1:app.gatv_data.data_length,app.gatv_setting.condition.dfnc_reponse,'--r','LineWidth',2);
                    ylim(app.UIAxes_Timeseries,'auto')
                else
                    plot(app.UIAxes_Timeseries,1:length(temp_data),temp_data,'b','LineWidth',2);
                    ylim(app.UIAxes_Timeseries,'auto')
                end
            else
                errordlg("Please make selection","Error");
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create GATFD_view_UIFigure and hide until all components are created
            app.GATFD_view_UIFigure = uifigure('Visible', 'off');
            app.GATFD_view_UIFigure.Position = [100 100 832 810];
            app.GATFD_view_UIFigure.Name = 'GAT-FD - Result Display v0.2a';
            app.GATFD_view_UIFigure.KeyPressFcn = createCallbackFcn(app, @GATFD_view_UIFigureKeyPress, true);

            % Create UIAxes_Network
            app.UIAxes_Network = uiaxes(app.GATFD_view_UIFigure);
            title(app.UIAxes_Network, 'Network')
            xlabel(app.UIAxes_Network, 'X')
            ylabel(app.UIAxes_Network, 'Y')
            app.UIAxes_Network.Position = [484 480 314 307];

            % Create FrameSliderLabel
            app.FrameSliderLabel = uilabel(app.GATFD_view_UIFigure);
            app.FrameSliderLabel.HorizontalAlignment = 'right';
            app.FrameSliderLabel.Position = [34 509 40 22];
            app.FrameSliderLabel.Text = 'Frame';

            % Create FrameSlider
            app.FrameSlider = uislider(app.GATFD_view_UIFigure);
            app.FrameSlider.ValueChangedFcn = createCallbackFcn(app, @gatv_update_network, true);
            app.FrameSlider.Position = [95 518 316 3];

            % Create LoadProcessedFileButton
            app.LoadProcessedFileButton = uibutton(app.GATFD_view_UIFigure, 'push');
            app.LoadProcessedFileButton.ButtonPushedFcn = createCallbackFcn(app, @gatv_load_file, true);
            app.LoadProcessedFileButton.Position = [67.5 754 124 22];
            app.LoadProcessedFileButton.Text = 'Load Processed File';

            % Create UIAxes_TaskDesign
            app.UIAxes_TaskDesign = uiaxes(app.GATFD_view_UIFigure);
            title(app.UIAxes_TaskDesign, 'Title')
            xlabel(app.UIAxes_TaskDesign, 'X')
            ylabel(app.UIAxes_TaskDesign, 'Y')
            app.UIAxes_TaskDesign.Position = [34 541 389 185];

            % Create UIAxes_Timeseries
            app.UIAxes_Timeseries = uiaxes(app.GATFD_view_UIFigure);
            title(app.UIAxes_Timeseries, 'Title')
            xlabel(app.UIAxes_Timeseries, 'X')
            ylabel(app.UIAxes_Timeseries, 'Y')
            app.UIAxes_Timeseries.Position = [53 42 745 239];

            % Create LoadTemporalMaskButton
            app.LoadTemporalMaskButton = uibutton(app.GATFD_view_UIFigure, 'push');
            app.LoadTemporalMaskButton.ButtonPushedFcn = createCallbackFcn(app, @gatv_load_design, true);
            app.LoadTemporalMaskButton.Position = [233.5 754 128 22];
            app.LoadTemporalMaskButton.Text = 'Load Temporal Mask';

            % Create LoadNetworkPropertyFileButton
            app.LoadNetworkPropertyFileButton = uibutton(app.GATFD_view_UIFigure, 'push');
            app.LoadNetworkPropertyFileButton.ButtonPushedFcn = createCallbackFcn(app, @gatv_load_measure, true);
            app.LoadNetworkPropertyFileButton.Position = [53 431 162 22];
            app.LoadNetworkPropertyFileButton.Text = 'Load Network Property File';

            % Create ThresholdDropDownLabel
            app.ThresholdDropDownLabel = uilabel(app.GATFD_view_UIFigure);
            app.ThresholdDropDownLabel.Position = [53 331 59 22];
            app.ThresholdDropDownLabel.Text = 'Threshold';

            % Create ThresholdDropDown
            app.ThresholdDropDown = uidropdown(app.GATFD_view_UIFigure);
            app.ThresholdDropDown.Items = {'Mean'};
            app.ThresholdDropDown.Position = [266 331 199 22];
            app.ThresholdDropDown.Value = 'Mean';

            % Create MeasureDropDownLabel
            app.MeasureDropDownLabel = uilabel(app.GATFD_view_UIFigure);
            app.MeasureDropDownLabel.Position = [53 362 52 22];
            app.MeasureDropDownLabel.Text = 'Measure';

            % Create MeasureDropDown
            app.MeasureDropDown = uidropdown(app.GATFD_view_UIFigure);
            app.MeasureDropDown.Items = {};
            app.MeasureDropDown.Position = [266 362 199 22];
            app.MeasureDropDown.Value = {};

            % Create SubjectDropDownLabel
            app.SubjectDropDownLabel = uilabel(app.GATFD_view_UIFigure);
            app.SubjectDropDownLabel.Position = [53 395 46 22];
            app.SubjectDropDownLabel.Text = 'Subject';

            % Create SubjectDropDown
            app.SubjectDropDown = uidropdown(app.GATFD_view_UIFigure);
            app.SubjectDropDown.Items = {'GroupAverage'};
            app.SubjectDropDown.Position = [266 395 199 22];
            app.SubjectDropDown.Value = 'GroupAverage';

            % Create UpdateButton
            app.UpdateButton = uibutton(app.GATFD_view_UIFigure, 'push');
            app.UpdateButton.ButtonPushedFcn = createCallbackFcn(app, @gatv_update_timeseries, true);
            app.UpdateButton.Position = [53 297 100 22];
            app.UpdateButton.Text = 'Update';

            % Create FrameSliderLabelCursor
            app.FrameSliderLabelCursor = uilabel(app.GATFD_view_UIFigure);
            app.FrameSliderLabelCursor.HorizontalAlignment = 'right';
            app.FrameSliderLabelCursor.Position = [437 509 25 22];
            app.FrameSliderLabelCursor.Text = '0';

            % Show the figure after all components are created
            app.GATFD_view_UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = gatfd_ui_view

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.GATFD_view_UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.GATFD_view_UIFigure)
        end
    end
end