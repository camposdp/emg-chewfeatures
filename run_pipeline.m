%% run_pipeline.m
clc; clear; close all;

fprintf('========================================\n');
fprintf('Step 1: Extracting features...\n');
try
    run('extract_features.m');
    fprintf('Feature extraction completed successfully!\n');
catch ME
    fprintf(2, 'Error during feature extraction: %s\n', ME.message);
    return;
end
fprintf('========================================\n\n');

fprintf('========================================\n');
fprintf('Step 2: Selecting features (forward selection)...\n');
try
    run('select_features.m');
    fprintf('Feature selection completed successfully!\n');
catch ME
    fprintf(2, 'Error during feature selection: %s\n', ME.message);
    return;
end
fprintf('========================================\n\n');

fprintf('========================================\n');
fprintf('Step 3: Evaluating models and generating figures...\n');
try
    run('evaluate_models.m');
    fprintf('Model evaluation completed successfully!\n');
catch ME
    fprintf(2, 'Error during model evaluation: %s\n', ME.message);
    return;
end
fprintf('========================================\n\n');

fprintf('✅ All steps completed! Check the /figures and /tables folders for outputs.\n');
fprintf('========================================\n');
