%将数据导出为用于训练的格式
%X  n*t
%Label  n*1
clear all
clc

%% 文件名匹配
Data_path = './hc3/';
cluFiles = dir(fullfile(Data_path, "*.clu.*"));

pairs = strings(0,2);
missing_spk = strings(0,1);

for i = 1:numel(cluFiles)
    cluName = string(cluFiles(i).name);
    spkName = replace(cluName, ".clu.", ".spk.");   

    spkPath = fullfile(Data_path, spkName);
    cluPath = fullfile(Data_path, cluName);

    if isfile(spkPath)
        pairs(end+1,:) = [cluPath, spkPath]; 
    else
        missing_spk(end+1) = cluPath; 
    end
end

T = table(pairs(:,1), pairs(:,2), 'VariableNames', {'clu_file','spk_file'});
disp(T)


%% 读取数据

Spike_wave = struct();

for k = 1:height(T)
    clu_file = T.clu_file(k);
    spk_file = T.spk_file(k);
    clu_data = load(clu_file);
    S_data = memmapfile(spk_file, 'Format', 'int16');
    data = S_data.Data;
    n_spk = length(data)/(32*8);
    if n_spk~=(length(clu_data)-1)
        continue
    end
    spk_data = reshape(data, [32, 8, length(data)/(32*8)]);
    c_n = clu_data(1);
    wave_data = {};
    for c = 1:c_n
        class_index = (clu_data(2:end,1) == c);
        class_spike = spk_data(:,:,class_index);
        wave_data{c} = class_spike;
    end
    Spike_wave(k).Name = spk_file;
    Spike_wave(k).Wave = wave_data;    
    fprintf("Pair %d:\n  %s\n  %s\n", k, clu_file, spk_file);
end



%% 储存
save('./Waveform.mat',"Spike_wave")


























