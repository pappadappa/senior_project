clear all
close all
clc

folder = 'D:\______PROJECT_____\Cut arduio_origin\Save_S_output_Crackle';
audio_files = dir(fullfile(folder,'*.mat'));

for k=1:length(audio_files)
    fileNames = load(audio_files(k).name); 
    fileNames_Save = strrep(audio_files(k).name,'.mat','') % eraser
    
    Fs = 44100 ;
    n = 2^nextpow2(3.3338*10e4);
    f = Fs*(0:(n/2))/n;
    Y = fft(fileNames.s1_output,n); 
    P = abs(Y/n);
    
%     figure
    plot(f,P(1:n/2+1)) 
    title(audio_files(k).name)
    xlabel('Frequency (f)')
    ylabel('|P(f)|')
    hold on
    
    
    save(['D:\______PROJECT_____\Cut arduio_origin\FFT_Nor_Crackle/P_FFT_' , fileNames_Save, '.mat'],'P')
    save(['D:\______PROJECT_____\Cut arduio_origin\FFT_Nor_Crackle/f_FFT_' , fileNames_Save, '.mat'],'f')
    
end