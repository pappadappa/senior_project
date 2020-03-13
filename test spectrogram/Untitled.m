clear all
close all
clc

folder = 'E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze';
audio_files = dir(fullfile(folder,'*.mat'));

for k=1:1
    fileNames = load(audio_files(k).name); 
    fileNames_Save = strrep(audio_files(k).name,'.mat','');
    
    s = spectrogram(fileNames.s1_output);
    spectrogram(fileNames.s1_output, 'yaxis');
    saveas(gcf,'filename.png');
end
