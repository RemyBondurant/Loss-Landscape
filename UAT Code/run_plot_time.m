clc
clear
close all

folderArray = {'C:\Users\Gambit\Desktop\UAT Code\mse\Adam(0.00001)_100';
    'C:\Users\Gambit\Desktop\UAT Code\mse\Adam(0.001)_100';
    'C:\Users\Gambit\Desktop\UAT Code\mse\SGD(0.001)_100';
    'C:\Users\Gambit\Desktop\UAT Code\mse\SGD(decay)_100';
    'C:\Users\Gambit\Desktop\UAT Code\mse\Adagrad(decay)_100'};
f1 = figure();
f2 = figure();
for i = 1:length(folderArray)
    [timeArray, mseArray] = plot_training_time(folderArray{i});
    figure(f1)
    plot(25:25:750, timeArray, 'Marker', 'diamond', 'LineStyle', '-')
    hold on
    grid on
    figure(f2)
    semilogy(25:25:750, mseArray, 'Marker', 'diamond', 'LineStyle', '-')
    hold on
    grid on
end
figure(f1)
legend('Adam, LR: 0.00001', 'Adam, LR: 0.001', 'SGD, LR: 0.001', 'SGD, LR: Decay', 'Adagrad, LR: Decay', 'Location', 'northwest')
xlabel('Number of Points')
ylabel('Training Time (s)')
figure(f2)
legend('Adam, LR: 0.00001', 'Adam, LR: 0.001', 'SGD, LR: 0.001', 'SGD, LR: Decay', 'Adagrad, LR: Decay')
xlabel('Number of Points')
ylabel('L2 Error')