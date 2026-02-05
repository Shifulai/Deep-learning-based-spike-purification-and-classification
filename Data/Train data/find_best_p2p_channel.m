function bestCh = find_best_p2p_channel(waveforms,mode)
% FIND_BEST_P2P_CHANNEL
% Automatically find the channel with the largest mean peak-to-peak value
%
% Input:
%   waveforms : T x C x N array
%       T = number of time samples
%       C = number of channels
%       N = number of spikes
%   mode      : 'maxmin'  -> max - min (default)
%               'minzero' -> 0 - min
%
% Output:
%   bestCh  : index (1-based) of channel with largest mean P2P
%   meanP2P : 1 x C vector, mean P2P value for each channel

    % sanity check
    if ndims(waveforms) ~= 3
        error('Input must be a T x C x N array.');
    end

    if nargin < 2 || isempty(mode)
        mode = 'maxmin';
    end

    % peak-to-peak for each spike and channel
    % result: C x N
    wave_average = mean(waveforms,3);

    switch mode
        case 'maxmin'
            % peak-to-peak
            p2p = squeeze(max(wave_average, [], 1) - min(wave_average, [], 1));

        case 'minzero'
            % distance from trough to zero
            p2p = abs(0 - min(wave_average, [], 1));

    end
    [~, bestCh] = max(p2p);
end
