clc;
clear;

% Stage 0: digital-twin -> real (McAllister), from the original pipeline.
% Saved shape: (4 metrics, 30 seeds, 21 points) -> average over dim 2 (seeds).
load('result/all_acc_train_on_transfer_measured.mat');
stage0_acc = squeeze(mean(all_acc_train_on_transfer_measured, 2));
stage0_points = 0:5:100;

% Stage 1: four real-to-real comparisons. Each file was saved from Python as
% {"acc": all_acc, "sweep_points": [0] + SWEEP_POINTS}, with acc shaped
% (4 metrics, points, seeds) -> average over dim 3 (seeds), NOT dim 2 like
% the Stage 0 file above, since train_model_cross_scenario.py stacks its
% axes in a different order than the original transfer-learning script.
comparisons = {
    'time_of_day_at_mcallister', '#0072BD', 'McAllister day -> night';
    'time_of_day_at_ruralroad',  '#4DBEEE', 'Rural Road day -> night';
    'site_during_day',           '#D95319', 'McAllister -> Rural Road (day)';
    'site_during_night',         '#EDB120', 'McAllister -> Rural Road (night)';
};

figure;
plot(stage0_points, stage0_acc(2, :) * 100, '--ko', 'LineWidth', 1.5, ...
    'DisplayName', 'Stage 0: digital-twin -> real (McAllister)');
hold on;

for i = 1:size(comparisons, 1)
    label = comparisons{i, 1};
    color = comparisons{i, 2};
    nice_name = comparisons{i, 3};

    data = load(['result/stage1_' label '_acc.mat']);
    acc = squeeze(mean(data.acc, 3));   % average over seeds (3rd dim) -> (4 metrics, points)
    points = double(data.sweep_points);

    plot(points, acc(2, :) * 100, '-o', 'Color', color, 'LineWidth', 1.5, ...
        'DisplayName', ['Stage 1: ' nice_name]);
    hold on;
end

yline(12.5, ':', 'Random chance (top-2 of 16 beams)', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2);

xlabel('Number of real target-domain samples used for fine-tuning');
ylabel('Top-2 beam prediction accuracy (%)');
title('Recovery curves: digital-twin gap (Stage 0) vs. real-world environmental shifts (Stage 1)');
legend('Location', 'eastoutside');
grid on;

if ~exist('final figures', 'dir')
    mkdir('final figures');
end

savefig(gcf, 'final figures/stage1_recovery_comparison.fig');
exportgraphics(gcf, 'final figures/stage1_recovery_comparison.png');
disp('Saved final figures/stage1_recovery_comparison.fig and .png');
