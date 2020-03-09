clear all
close all
clc

folder = 'E:\database\ICBHI_final_database\pass normalize\Save_S_output_Crackle';
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
    
    
    save(['E:\database\ICBHI_final_database\pass normalize\Save_S_output_Crackle\FFT\P_FFT_' , fileNames_Save, '.mat'],'P')
    save(['E:\database\ICBHI_final_database\pass normalize\Save_S_output_Crackle\FFT\f_FFT_' , fileNames_Save, '.mat'],'f')
    
end