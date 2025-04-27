clc; clear;

baseDir    = './raw_data';
outputDir  = './figuras_segmentos';
if ~exist(outputDir,'dir'), mkdir(outputDir); end

categories = {'Capim Estrela Africana','Capim Kurumi','Feno'};
XE = struct(); XD = struct(); XB = struct();

% Parâmetros de processamento
fs     = 1000;      % Hz
Wenv   = 50;        % janela RMS móvel (amostras)
Wseg   = 50;        % janela de limpeza da binarização
h_sigma= 3;         % limiar = mu + h_sigma * sigma

for c = 1:numel(categories)
    catName = categories{c};
    catKey  = lower(strrep(catName,' ','')); % chave sem espaço
    files   = dir(fullfile(baseDir,catName,'*.mat'));
    
    for f = files'
        data = load(fullfile(baseDir,catName,f.name));
        if ~isfield(data,'CH1') || ~isfield(data,'CH2'), continue; end
        
        repNum = regexp(f.name,'\d+','match','once');
        repKey = ['rep',repNum];
        
        % --- 1) Filtrar sinais ---
        sig1 = filtroEMG(data.CH1);
        sig2 = filtroEMG(data.CH2);
        
        % --- 2) Envelope RMS móvel ---
        env1 = sqrt(movmean(sig1.^2, Wenv));
        env2 = sqrt(movmean(sig2.^2, Wenv));
        
        % --- 3) Detecção individual (para XE e XD) ---
        T1 = mean(env1) + h_sigma * std(env1);
        T2 = mean(env2) + h_sigma * std(env2);
        bin1 = movmean(env1 > T1, Wseg) > 0.5;
        bin2 = movmean(env2 > T2, Wseg) > 0.5;
        
        % --- 4) Janelas individuais ---
        onset1 = find(diff([0;bin1])>0);
        offset1= find(diff([bin1;0])<0);
        onset2 = find(diff([0;bin2])>0);
        offset2= find(diff([bin2;0])<0);
        
        % --- 5) Detecção combinada pelo envelope médio ---
        env_mean = (env1 + env2) / 2;
        T_mean   = mean(env_mean) + h_sigma * std(env_mean);
        binB     = movmean(env_mean > T_mean, Wseg) > 0.5;
        onsetB   = find(diff([0;binB])>0);
        offsetB  = find(diff([binB;0])<0);
        
        % --- 6) Extração de features ---
        XE.(catKey).(repKey) = extract_all_features(sig1, onset1, offset1);
        XD.(catKey).(repKey) = extract_all_features(sig2, onset2, offset2);
        XB.(catKey).(repKey).left  = extract_all_features(sig1, onsetB, offsetB);
        XB.(catKey).(repKey).right = extract_all_features(sig2, onsetB, offsetB);
        
        % --- 7) (Opcional) Plot de verificação ---
        plot_and_save(env1, bin1, fullfile(outputDir,[repKey,'_CH1.png']), 'Esquerdo');
        plot_and_save(env2, bin2, fullfile(outputDir,[repKey,'_CH2.png']), 'Direito');
        plot_and_save(env_mean, binB, fullfile(outputDir,[repKey,'_AMBOS.png']), 'Envelope Médio');
    end
end

% --- 8) Salvar ---
save('featuresData_XE.mat','XE');
save('featuresData_XD.mat','XD');
save('featuresData_XB.mat','XB');

disp('Extração concluída!');

%% Funções Auxiliares

function Y = filtroEMG(X)
    fs = 1000;
    [B,A] = butter(4, 20/(fs/2), 'high');
    Y = filtfilt(B,A,X);
end

function F = extract_all_features(sig, onset, offset)
    % Pacote de 9 features
    N = min(numel(onset),numel(offset));
    F = nan(9,N);
    [BD,IchT,CD] = compute_time_feats(onset,offset);
    for i = 1:N
        seg = sig(onset(i):offset(i));
        RMS  = rms(seg);
        MAV  = mean(abs(seg));
        ZC   = sum(abs(diff(sign(seg)))==2);
        WL   = sum(abs(diff(seg)));
        SSC  = sum(diff(sign(diff(seg)))~=0);
        IEMG = sum(abs(seg));
        F(:,i) = [RMS; MAV; ZC; WL; SSC; IEMG; BD(i); IchT(i); CD(i)];
    end
end

function [BD,IchT,CD] = compute_time_feats(onset,offset)
    N = min(numel(onset),numel(offset));
    BD   = nan(1,N);
    IchT = nan(1,N);
    CD   = nan(1,N);
    for i = 1:N-1
        BD(i)   = offset(i) - onset(i);
        IchT(i) = onset(i+1) - offset(i);
        CD(i)   = onset(i+1) - onset(i);
    end
    if N >= 1
        BD(N) = offset(N) - onset(N);
    end
end

function plot_and_save(env, bin, filepath, label)
    figure('Visible','off');
    t = 1:length(env);
    plot(t, env, 'k'); hold on;
    patchIdx = find(bin);
    if ~isempty(patchIdx)
        area(t(patchIdx), env(patchIdx), 'FaceColor',[1 0.8 0.8],'EdgeColor','none');
    end
    title(label); xlabel('Amostras'); ylabel('Envelope RMS');
    grid on;
    saveas(gcf,filepath);
    close;
end
