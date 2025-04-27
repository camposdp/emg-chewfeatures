clc; clear; close all;

%% 1) Output folders
mkdir('figures/forward_selection');
mkdir('figures/bar_performance');
mkdir('figures/confusion_matrices');
mkdir('tables');

%% 2) Load feature structures
load('featuresData_XE.mat','XE');
load('featuresData_XD.mat','XD');
load('featuresData_XB.mat','XB');

categories = {'capimestrelaafricana','capimkurumi','feno'};
sides = {'XE','XD','Both'};
baseFeatures = {'RMS','MAV','ZC','WL','SSC','IEMG','BD','IchT','CD'};
methods = {'SVM_RBF','LDA','SVM_LINEAR','KNN'};

accCurves = struct();
featOrders = struct();
foldAccuracies = struct();
trueLabels = struct();
predLabels = struct();
bestSelections = struct();

%% 3) Forward Selection and Evaluation
for m = 1:numel(methods)
    method = methods{m};
    
    for si = 1:numel(sides)
        side = sides{si};
        fprintf('--- %s on %s ---\n', method, side);
        
        % Prepare data
        [X, y] = prepareEMGData(side, XE, XD, XB, categories);
        [Xb, yb] = balance_classes(X, y);
        
        % Define feature groups
        F = numel(baseFeatures);
        if strcmp(side,'Both')
            groups = arrayfun(@(i)[i,i+F], 1:F, 'UniformOutput', false);
        else
            groups = num2cell(1:F);
        end
        
        % Forward selection
        [selectedGroups, accs, orders] = forwardGroupedFS(method, Xb, yb, groups, baseFeatures);
        
        accCurves.(method).(side) = accs;
        featOrders.(method).(side) = orders;
        
        % Find best step (highest mean accuracy)
        [~, bestStep] = max(accs);
        bestSelections.(method).(side).groups = selectedGroups(1:bestStep);
        bestSelections.(method).(side).order = orders(1:bestStep);
        
        % Evaluate using only best-selected features
        bestIdx = cell2mat(groups(bestSelections.(method).(side).groups));
        
        cvp = cvpartition(yb,'KFold',5);
        accFolds = zeros(cvp.NumTestSets,1);
        yTrue = []; yPred = [];
        
        for k = 1:cvp.NumTestSets
            trainIdx = training(cvp,k);
            testIdx = test(cvp,k);
            switch method
                case 'SVM_RBF'
                    mdl = fitcecoc(Xb(trainIdx,bestIdx), yb(trainIdx), 'Learners', templateSVM('KernelFunction','rbf'));
                case 'LDA'
                    mdl = fitcdiscr(Xb(trainIdx,bestIdx), yb(trainIdx));
                case 'SVM_LINEAR'
                    mdl = fitcecoc(Xb(trainIdx,bestIdx), yb(trainIdx), 'Learners', templateSVM('KernelFunction','linear'));
                case 'KNN'
                    mdl = fitcknn(Xb(trainIdx,bestIdx), yb(trainIdx), 'NumNeighbors',5);
            end
            preds = predict(mdl, Xb(testIdx,bestIdx));
            accFolds(k) = mean(preds == yb(testIdx));
            yTrue = [yTrue; yb(testIdx)];
            yPred = [yPred; preds];
        end
        
        foldAccuracies.(method).(side) = accFolds;
        trueLabels.(method).(side) = yTrue;
        predLabels.(method).(side) = yPred;
        
        fprintf('  Best combination: %d features — Mean ± Std accuracy = %.1f%% ± %.1f%%\n', ...
                bestStep, mean(accFolds)*100, std(accFolds)*100);
    end
end

%% 4) Save Raw Results
save('tables/results_raw.mat', 'accCurves', 'featOrders', 'foldAccuracies', ...
    'trueLabels', 'predLabels', 'bestSelections', 'methods', 'sides', 'baseFeatures');

clc; clear; close all;

%% 1) Load raw results
load('tables/results_raw.mat');

% Create folders if not exist
mkdir('figures/forward_selection');
mkdir('figures/bar_performance');
mkdir('figures/confusion_matrices');

% Set colorblind-friendly colors
colors = [
    0 114 189;    % blue - XE
    217 83 25;    % orange - XD
    119 172 48    % green - Both
    ] / 255;

sidesLong = {'Left Side (XE)','Right Side (XD)','Both Sides'};

%% 2) Plot Forward Selection curves (all methods together)
figure('Units','normalized','Position',[0.2 0.2 1 0.8]);
tiledlayout(2,2,'TileSpacing','compact');

for m = 1:numel(methods)
    nexttile;
    hold on;
    method = methods{m};
    for si = 1:numel(sides)
        side = sides{si};
        a = accCurves.(method).(side)*100;
        plot(1:numel(a), a, '-o', ...
             'Color', colors(si,:), 'MarkerFaceColor', colors(si,:), ...
             'LineWidth', 1.8, 'MarkerSize',6);
    end
    xlabel('Number of Selected Features','FontWeight','bold');
    ylabel('Accuracy (%)','FontWeight','bold');
    title(strrep(method,'_','-'),'FontWeight','bold');
    xticks(1:9); xlim([0.5 9.5]);
    ylim([30 80]);
    grid on;
    if m==1
        
        legend(sidesLong,'Location','southoutside','Orientation','horizontal');
    end
    hold off;
end

exportgraphics(gcf, 'figures/forward_selection/forward_selection_all.pdf', ...
    'ContentType','vector','BackgroundColor','none');

%% 3) Plot bar charts ± std (all methods)
figure('Units','normalized','Position',[0.2 0.2 1 0.8]);
tiledlayout(2,2,'TileSpacing','compact');

for m = 1:numel(methods)
    nexttile;
    method = methods{m};
    means = cellfun(@(s) mean(foldAccuracies.(method).(s)), sides);
    stds  = cellfun(@(s) std(foldAccuracies.(method).(s)), sides);
    
    bh=bar(1:3, means*100, 'FaceColor','flat');
    hold on;
    errorbar(1:3, means*100, stds/sqrt(length(stds))*100, '.k','LineWidth',1.2,'CapSize',10);
    set(gca,'XTick',1:3,'XTickLabel',sidesLong);
    ylabel('Accuracy (%)','FontWeight','bold');
    ylim([85 100]);
    title(strrep(method,'_','-'),'FontWeight','bold');
    grid on;

     
    bh.CData(1,:) = colors(1,:);
    bh.CData(2,:) = colors(2,:);
    bh.CData(3,:) = colors(3,:);
end

exportgraphics(gcf, 'figures/bar_performance/bar_performance_all.pdf', ...
    'ContentType','vector','BackgroundColor','none');

%% 4) Confusion Matrices for the Best Side
for m = 1:numel(methods)
    method = methods{m};
    vals = cellfun(@(s) mean(foldAccuracies.(method).(s)), sides);
    [~, idx] = max(vals);
    bestSide = sides{idx};
    
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
    confusionchart(trueLabels.(method).(bestSide), predLabels.(method).(bestSide), ...
                   'Title',['Confusion Matrix - ' strrep(method,'_','-') ' (' bestSide ')']);
    exportgraphics(gcf, sprintf('figures/confusion_matrices/confusion_%s_%s.pdf',method,bestSide), ...
        'ContentType','vector','BackgroundColor','none');
end

%% 5) Save Selected Features Table
disp('Saving selected features table...');
writetable_features(bestSelections, methods, sides, 'tables/selected_features.xlsx');

%% 6) Save Accuracy Progression Table
disp('Saving accuracy progression table...');
writetable_progression(accCurves, featOrders, methods, sides, 'tables/accuracy_progression.xlsx');

%% 7) Perform ANOVA Statistics
disp('Performing ANOVA...');
Tstats = table();
for m = 1:numel(methods)
    method = methods{m};
    XE_acc = foldAccuracies.(method).('XE');
    XD_acc = foldAccuracies.(method).('XD');
    Both_acc = foldAccuracies.(method).('Both');
    allData = [XE_acc; XD_acc; Both_acc];
    group = [repmat({'Left'},numel(XE_acc),1); ...
             repmat({'Right'},numel(XD_acc),1); ...
             repmat({'Both'},numel(Both_acc),1)];
    p = anova1(allData, group, 'off');
    
    % Store
    Tstats = [Tstats; table({strrep(method,'_','-')}, p, ...
              mean(XE_acc)*100, mean(XD_acc)*100, mean(Both_acc)*100, ...
              'VariableNames', {'Method','p_value','Mean_Left','Mean_Right','Mean_Both'})];
end

disp(Tstats);
writetable(Tstats, 'tables/statistics_summary.xlsx');

%% ====== Auxiliary Functions (Evaluation/Export) ======

function writetable_features(bestSelections, methods, sides, filename)
    % Create a table listing selected features for each method and side
    T = table();
    for m = 1:numel(methods)
        method = methods{m};
        for si = 1:numel(sides)
            side = sides{si};
            if isfield(bestSelections.(method), side)
                feats = bestSelections.(method).(side).order;
                varname = sprintf('%s_%s', strrep(method,'_','-'), side);
                featsStr = join(string(feats), ', ');
                T.(varname) = featsStr;
            end
        end
    end
    writetable(T, filename, 'WriteVariableNames', true);
end

function writetable_progression(accCurves, featOrders, methods, sides, filename)
    % Create a table showing accuracy progression with features
    varNames = {};
    Tcell = {};
    
    for m = 1:numel(methods)
        method = methods{m};
        for si = 1:numel(sides)
            side = sides{si};
            accs = accCurves.(method).(side) * 100; % in percentage
            feats = featOrders.(method).(side);
            
            if isempty(accs)
                continue;
            end
            
            method_side = sprintf('%s_%s', strrep(method,'_','-'), side);
            
            for k = 1:numel(accs)
                Tcell = [Tcell; {method_side, k, feats{k}, accs(k)}]; %#ok<AGROW>
            end
        end
    end
    
    T = cell2table(Tcell, 'VariableNames', {'Method_Side','NumFeatures','FeatureAdded','Accuracy'});
    writetable(T, filename, 'WriteVariableNames', true);
end


%% ====== Funções Auxiliares ======

function [selG, accs, order] = forwardGroupedFS(method, X, y, groups, featNames)
    rem = 1:numel(groups); 
    sel = []; 
    accs = []; 
    order = {};
    while ~isempty(rem)
        bestAcc = -inf;
        bestG   = NaN;
        for g = rem
            idxs = cell2mat(groups([sel g]));
            switch method
                case 'SVM_RBF'
                    mdl = fitcecoc(X(:,idxs), y, 'Learners', templateSVM('KernelFunction','rbf'));
                case 'LDA'
                    mdl = fitcdiscr(X(:,idxs), y);
                case 'SVM_LINEAR'
                    mdl = fitcecoc(X(:,idxs), y, 'Learners', templateSVM('KernelFunction','linear'));
                case 'KNN'
                    mdl = fitcknn(X(:,idxs), y, 'NumNeighbors',5);
            end
            cvmdl = crossval(mdl, 'KFold', 5);
            acc = 1 - kfoldLoss(cvmdl);
            if acc > bestAcc
                bestAcc = acc;
                bestG   = g;
            end
        end
        sel(end+1)      = bestG;          %#ok<AGROW>
        rem(rem==bestG) = [];
        accs(end+1)     = bestAcc;         %#ok<AGROW>
        order{end+1}    = featNames{bestG}; %#ok<AGROW>
    end
    selG = sel;
end

function [X, y] = prepareEMGData(side, XE, XD, XB, cats)
    X = []; y = [];
    for ci = 1:numel(cats)
        cat = cats{ci};
        reps = fieldnames(XE.(cat));
        for r = 1:numel(reps)
            rep = reps{r};
            switch side
                case 'XE'
                    M = XE.(cat).(rep);  % 9×Ni
                case 'XD'
                    M = XD.(cat).(rep);  % 9×Ni
                otherwise % Both
                    M = [XB.(cat).(rep).left; XB.(cat).(rep).right]; % 18×Ni
            end
            X = [X; M'];                        % Linhas = segmentos
            y = [y; repmat(ci, size(M,2),1)];    % 1 = Capim Estrela, 2 = Feno
        end
    end
end

function [Xb, yb] = balance_classes(X, y)
    classes = unique(y);
    counts  = arrayfun(@(c) sum(y==c), classes);
    nmin    = min(counts);
    Xb = []; yb = [];
    for c = classes'
        idx = find(y==c);
        sel = idx(randperm(numel(idx), nmin));
        Xb  = [Xb; X(sel,:)]; 
        yb  = [yb; y(sel)];
    end
end

