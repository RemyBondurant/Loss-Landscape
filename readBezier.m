clc
close all
clear

% Specify the folder where the files live.
myFolder = 'checkpoints';
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', myFolder);
end
% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(myFolder, '*.mat'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
max_bez_array = [];
mean_bez_array = [];
max_line_array = [];
mean_line_array = [];
ratio_array = [];
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', baseFileName);
    load(fullFileName)
    figure()
    plot(t, bezier_data)
    hold on
    plot(t, straight_data)
    legend('Bezier Curve', 'Straight Segments')
    xlabel('t')
    ylabel('Train Error (L2)')
    title('Train Error Along Path For Two Minimums')
    
    m1 = bezier_data(1);
    m2 = bezier_data(1001);
    average_m = (m1 + m2) / 2;

    max_bez = max(bezier_data);
    
    ratio = max_bez / average_m;
    ratio_array = [ratio_array, ratio];

    max_bez_array = [max_bez_array, max_bez];
    % min_bez = min(bezier_data);
    mean_bez = mean(bezier_data);
    mean_bez_array = [mean_bez_array, mean_bez];

    max_line = max(straight_data);
    max_line_array = [max_line_array, max_line];
    % min_line = min(straight_data);
    mean_line = mean(straight_data);
    mean_line_array = [mean_line_array, mean_line];
    
    fprintf('Maximum Train Error: %0.4d (Bezier) vs %0.4d (Straight Line)\n', max_bez, max_line)
    fprintf('Average Train Error: %0.4d (Bezier) vs %0.4d (Straight Line)\n', mean_bez, mean_line)
end

final_array = [max_bez_array; mean_bez_array; max_line_array; mean_line_array];