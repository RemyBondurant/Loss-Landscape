clc
clear
close all


folderArray = {'C:\Users\Gambit\Desktop\UAT Code\mse\SGD(0.001)_1000\func_000750';
               'C:\Users\Gambit\Desktop\UAT Code\mse\SGD(decay)_1000\func_000750';
               'C:\Users\Gambit\Desktop\UAT Code\mse\Adam(decay)_1000\func_000750'};
for i = 1:length(folderArray)
    myFolder = folderArray{i};
    filePattern = fullfile(myFolder, '*.mat'); % Change to whatever pattern you need.
    theFiles = dir(filePattern);
    figure();
    for k = 1 : length(theFiles)
        clear mse epoch_times
        baseFileName = theFiles(k).name;
        fullFileName = fullfile(theFiles(k).folder, baseFileName);
        % fprintf(1, 'Now reading %s\n', fullFileName);
        load(fullFileName)
        semilogy(mse)
        hold on
    end
    xlabel('Epoch')
    ylabel('Training Loss')
    % ylim([10^(-2), 10^(0)])
    grid on
    switch i
        case 1
            title('Training Loss for Each Trial (SGD, LR: 0.001, 750 Points)')
        case 2
            title('Training Loss for Each Trial (SGD, LR: Decaying, 750 Points)')
        case 3
            title('Training Loss for Each Trial (Adam, LR: Decaying, 750 Points)')
    end

end