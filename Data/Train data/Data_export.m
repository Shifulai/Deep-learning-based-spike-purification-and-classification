%查看数据以及导出数据
clear all
clc

%% 读取数据
Data = load('./Waveform.mat');
Wave_data = Data.Spike_wave;

%% 绘制数据
C = [1,2];
class = 3;
xvar = 1:32;

for i = 1:length(C)
    data_cell = Wave_data(C(i)).Wave;
    data = data_cell{class};
    bestCh = find_best_p2p_channel(data,'minzero');
    figure
    for chanel_i = 1:8
        yvar = squeeze(data(:,chanel_i,:));
        yvar_clip = yvar(:,1:500);
        subplot(8,1,chanel_i)
        plot(xvar,yvar_clip,'Color', [0.5 0.5 0.5])
        hold on
        if chanel_i == bestCh
            plot(xvar,mean(yvar,2),'r')
        else
            plot(xvar,mean(yvar,2),'k')
        end
    end
end



%% 提取训练用X，Y
n = length(Wave_data);
Train_X = [];
Train_Y = [];
for i1 = 1:n
    data_cell = Wave_data(i1).Wave;
    for i2 = 1:length(data_cell)
        wave_data = data_cell{i2};
        if ndims(wave_data) ~= 3
            continue
        end
        bestCh = find_best_p2p_channel(wave_data,'minzero');
        %保留前10个sample
        x = squeeze(wave_data(1:10,chanel_i,:));
        if i2 == 1
            label_i = 0;
        else
            label_i = 1;
        end
        y = label_i*ones([size(x',1),1]);
        Train_X = [Train_X;x'];
        Train_Y = [Train_Y;y];
    end
end

%% 
n = length(Train_Y);

%训练集
Train_n = floor(n/3*2);
Train_X = Train_X(1:Train_n,:);
Train_Y = Train_Y(1:Train_n,:);
save('./Train_data.mat', 'Train_X', 'Train_Y', '-v7.3');

%测试集

Test_X = Train_X(Train_n+1:end,:);
Test_Y = Train_Y(Train_n+1:end,:);
save('./Test_data.mat', 'Test_X', 'Test_Y', '-v7.3');

