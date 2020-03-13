clear all
close all
clc

folder = 'E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze';
audio_files = dir(fullfile(folder,'*.mat'));

for k=1:length(audio_files)
    fileNames = load(audio_files(k).name); 
    fileNames_Save = strrep(audio_files(k).name,'.mat',''); % eraser
    
    
    Fs = 44100 ;
    disp(length(fileNames.s1_output));
    n = 2^nextpow2(length(fileNames.s1_output));
    f = Fs*(0:(n/2))/n;
    Y = fft(fileNames.s1_output,n); 
    P = abs(Y/n);
    
%     figure
    plot(f,P(1:n/2+1)) 
    hold on
    title(audio_files(k).name)
    xlabel('Frequency (f)')
    ylabel('|P(f)|')
    
    
    save(['E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze\power\P_FFT_' , fileNames_Save, '.mat'],'P')
    save(['E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze\frequency\f_FFT_' , fileNames_Save, '.mat'],'f')
end    
