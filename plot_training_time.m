function [timeArray, mseArray] = plot_training_time(targetFolder)

myFolder = targetFolder;
filePattern = fullfile(myFolder); % Change to whatever pattern you need.
theFiles = dir(filePattern);
timeArray = [];
mseArray = [];
for k = 3 : length(theFiles)
    currentTime = 0;
    currentMSE = 0;
    nextFolder = theFiles(k).name;
    currentFolder = append(theFiles(k).folder, '\', nextFolder);
    % disp(currentFolder)
    theFilesInFolder = dir(currentFolder);
    for j = 3 : length(theFilesInFolder)
        load(append(theFilesInFolder(j).folder, '\', theFilesInFolder(j).name), 'epoch_times', 'mse')
        % disp(sum(epoch_times))
        currentTime = currentTime + sum(epoch_times);
        currentMSE = currentMSE + min(mse);
    end
    % disp(currentTime)
    timeArray = [timeArray, currentTime];
    mseArray = [mseArray, currentMSE / 20];
end