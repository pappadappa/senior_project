folder1 = 'E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle\power';
P_files = dir(fullfile(folder1,'*.mat'));
folder2 = 'E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle\frequency';
F_files = dir(fullfile(folder2,'*.mat'));

FFT_all = [];

for k=1:length(P_files)
    F_fileNames = load(F_files(k).name); 
    F_fileNames_Save = strrep(F_files(k).name,'.mat','');
    
    P_fileNames = load(P_files(k).name); 
    P_fileNames_Save = strrep(P_files(k).name,'.mat','');
    
    n = 2^nextpow2(3.3338*10e4);
    
    plot(F_fileNames.f,P_fileNames.P(1:n/2+1))
    xlabel('Frequency (f)')
    ylabel('|P(f)|')
    
%     for i = 1:length(F_fileNames.f)
%         for j = 1:n/2+1 
%             
%             FFT_all(F_fileNames.f(i),:) = P_fileNames.P(j);
%             
%         end
%     end
    
end
