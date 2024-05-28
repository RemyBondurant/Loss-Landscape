clc;
close all;

folder_name = 'training_data';
if ~exist(folder_name, 'dir')
    mkdir(folder_name);
end

rng("default")
for n = 25:25:750
    x = -1 + 2 * rand(n, 20, "double");
    y = log10(sin(10*x)+2)+sin(x);
    file_name = "func_" + sprintf("%06d", n);
    filepath = fullfile(folder_name, file_name);
    save(filepath, 'x', 'y');
    % figure()
    % plot(x(:,1), y(:,1), 'or')
end
