%% select_features.m
clc; clear; close all;

%% 1) Carregar structs
load('featuresData_XE.mat','XE');
load('featuresData_XD.mat','XD');
load('featuresData_XB.mat','XB');

cats      = {'capimestrelaafricana','capimkurumi','feno'};
sides     = {'XE','XD','Both'};
baseFeat  = {'RMS','MAV','ZC','WL','SSC','IEMG','BD','IchT','CD'};

%% 2) Containers
results.SVM = struct();
results.LDA = struct();

%% 3) Loop Side × Método
for si = 1:numel(sides)
    side = sides{si};
    
    % 3.1) Montar X,y para esta side
    [X,y] = prepareEMGData(side, XE, XD, XB, cats);
    [Xb,yb] = balance_classes(X,y);
    
    % 3.2) Definir “grupos” de features
    F = numel(baseFeat);
    if strcmp(side,'Both')
        groups = arrayfun(@(i)[i, i+F], 1:F, 'UniformOutput',false);
        featNames = [ strcat(baseFeat,'_L'), strcat(baseFeat,'_R') ];
    else
        groups = num2cell(1:F);
        featNames = baseFeat;
    end
    
    % 3.3) Forward‐selection SVM
    [selSVM, accSVM, orderSVM] = forwardGroupedFS_SVM(Xb, yb, groups, featNames);
    results.SVM.(side).accuracies   = accSVM;
    results.SVM.(side).featureOrder = orderSVM;
    
    % 3.4) Forward‐selection LDA
    [selLDA, accLDA, orderLDA] = forwardGroupedFS_LDA(Xb, yb, groups, featNames);
    results.LDA.(side).accuracies   = accLDA;
    results.LDA.(side).featureOrder = orderLDA;
end

%% 4) Salvar resultados
save('results.mat','results');
fprintf('Tudo salvo em results.mat\n');


%% Funções auxiliares

function [X,y] = prepareEMGData(side, XE, XD, XB, cats)
    X=[]; y=[];
    for ci=1:2
        cat = cats{ci};
        reps = fieldnames(XE.(cat));
        for ri=1:numel(reps)
            rep = reps{ri};
            switch side
                case 'XE', M = XE.(cat).(rep);
                case 'XD', M = XD.(cat).(rep);
                otherwise
                    Ml = XB.(cat).(rep).left;
                    Mr = XB.(cat).(rep).right;
                    M  = [Ml; Mr];
            end
            X = [X; M'];        %#ok<AGROW>
            y = [y; repmat(ci,size(M,2),1)]; %#ok<AGROW>
        end
    end
end

function [Xb,yb] = balance_classes(X,y)
    cls = unique(y);
    n   = arrayfun(@(c)sum(y==c),cls);
    m   = min(n);
    Xb=[]; yb=[];
    for c0=cls'
        idx = find(y==c0);
        sel = idx(randperm(numel(idx),m));
        Xb  = [Xb; X(sel,:)]; 
        yb  = [yb; y(sel)];
    end
end

function [selG, accs, order] = forwardGroupedFS_SVM(X,y,groups,featNames)
    rem = 1:numel(groups); sel=[]; accs=[]; order={};
    while ~isempty(rem)
        bestAcc=-inf; bestG=NaN;
        for g=rem
            cols = [groups{[sel g]}];
            mdl  = fitcecoc(X(:,cols), y, 'Learners',templateSVM('KernelFunction','rbf'));
            cvm  = crossval(mdl,'KFold',5);
            a    = 1 - kfoldLoss(cvm);
            if a>bestAcc, bestAcc=a; bestG=g; end
        end
        sel(end+1)=bestG; rem(rem==bestG)=[]; accs(end+1)=bestAcc; order{end+1}=featNames{bestG}; %#ok<AGROW>
    end
    selG = sel;
end

function [selG, accs, order] = forwardGroupedFS_LDA(X,y,groups,featNames)
    rem = 1:numel(groups); sel=[]; accs=[]; order={};
    while ~isempty(rem)
        bestAcc=-inf; bestG=NaN;
        for g=rem
            cols = [groups{[sel g]}];
            mdl  = fitcdiscr(X(:,cols), y);
            cvm  = crossval(mdl,'KFold',5);
            a    = 1 - kfoldLoss(cvm);
            if a>bestAcc, bestAcc=a; bestG=g; end
        end
        sel(end+1)=bestG; rem(rem==bestG)=[]; accs(end+1)=bestAcc; order{end+1}=featNames{bestG}; %#ok<AGROW>
    end
    selG = sel;
end
