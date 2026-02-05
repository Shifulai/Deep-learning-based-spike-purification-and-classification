%数据读取并储存为需要的格式
clear all
clc

%% 
Data_path = './pl2 data/';
Data_dir = dir(Data_path);

for i1 = 3:length(Data_dir)
    Data_name = Data_dir(i1).name;
    data_path = fullfile(Data_path,Data_name);
    [~, name_channel] = plx_adchan_names(data_path);
    isWB = ~cellfun('isempty', regexp(cellstr(name_channel), 'WB'));
    channels_WB = name_channel(isWB, :);
    Raw = [];
    for i2 = 1:length(channels_WB)
        [ad_freq, ~, ~, ~, ad] = plx_ad(data_path,channels_WB(i2,:));
        raw_data = ad';
        if length(raw_data) ~= 1
           Raw(i2,:) = raw_data ;
           
        else
           continue
        end
    end
    write_path = fullfile('./',[Data_name '.mat']);
    save(write_path, 'Raw', 'ad_freq', '-v7.3');
end


        























