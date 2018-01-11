clc;
clear;

path = '/media/xnat/WinE/QSM_MNI/';
subjectList = dir(path);
dataPath = '/home/xnat/PD/QSMdata/';

for i = 3:length(subjectList)
    subjectName = subjectList(i).name;
    subjectPath = strcat(path, subjectName);
%     subjectSegList = dir(subjectPath);
%     for j = 3:length(subjectSegList)
%         if(strcmp(subjectSegList(j).name(1:2),'w2'))
            data = spm_read_vols(spm_vol(subjectPath));
            save(strcat(dataPath, subjectName(1:19)),'data'); 
%         end
%     end
end