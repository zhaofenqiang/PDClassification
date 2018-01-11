clc;
clear;

path = '/media/xnat/WinE/QSM_MNI/';
subjectList = dir(path);
dataPath = '/home/xnat/PD/patch3d/';

for i = 3:length(subjectList)
    subjectName = subjectList(i).name;
    data = spm_read_vols(spm_vol(strcat(path, subjectName)));
    patch = data(37:150, 62:165, 55:111);
    save(strcat(dataPath, subjectName(1:19)), 'patch'); 
end