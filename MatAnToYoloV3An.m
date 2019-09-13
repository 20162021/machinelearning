D = uigetdir(pwd, 'Select a folder with annotations');
K = uigetdir(pwd, 'Where to save modificated files');
cd (D)
number = numel(dir('*.mat'));
for ii = 1: number
    groundTruth = importdata(['V00' num2str(ii-1) '_lwir' '.mat']);
    DD = join(["V00",convertCharsToStrings(int2str(ii-1))],"");
    DDC = convertStringsToChars(DD);
    cd (K)
    mkdir (DDC)
    cd (D)
    for i=1:size(groundTruth.LabelData.pedestrian)
        chr = convertCharsToStrings(int2str(i-1));
           if i<10
               extra = "\I0000";
           elseif i>=10 && i<100
                   extra = "\I000";
           elseif i>=100 && i<1000
                   extra = "\I00";
           elseif i>=1000 && i<10000
                   extra = "\I0";
           elseif i>=10000 && i<100000
                   extra = "\I";
           end
           K = convertCharsToStrings(K);
           PH = [K,"\",DD,extra,chr,".txt"];
           PH = join(PH,"");
           PH = convertStringsToChars(PH);
       filePh = fopen(PH,'w');
       data = groundTruth.LabelData.pedestrian{i,:};
       if size(data)>0
           for j=1:size(data)
           data(j,1)=data(j,1)/640;
           data(j,2)=data(j,2)/512;
           data(j,3)=data(j,3)/640;
           data(j,4)=data(j,4)/512;
           end
        fprintf(filePh,'0 %1.2f %1.2f %1.2f %1.2f\n',data);
       end
       fclose(filePh);
     end
end