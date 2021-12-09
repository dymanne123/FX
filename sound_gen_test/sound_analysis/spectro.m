%%% bruitages starwars :
%%%    https://www.sound-fishing.net/sons/star-wars
%%%    http://www.universal-soundbank.com/starwars.htm

close all;
clear all;

pkg load signal

%[xxx,Fs]=audioread('SFB-sabre-Laser-02-SF.wav');
[xxx,Fs]=audioread('SFB-sabre-Laser-01-SF.wav');
xx=xxx(:,1);

step = fix(5*Fs/1000);         # one spectral slice every 3 ms
window = fix(40*Fs/1000);      # 40 ms data window
fftn = 2^nextpow2(window)*2^7; # next highest power of 2, times a power of 2 (possibly big)
[S1, f, t] = specgram(xx, fftn, Fs, window, window-step);
%S = abs(S1(2:fftn*4000/Fs,:));  # magnitude in range 0<f<=4000 Hz.
S = abs(S1);
S = S/max(S(:));               # normalize magnitude so that max is 0 dB.
S = max(S, 10^(-40/10));       # clip below -40 dB.
S = min(S, 10^(-3/10));        # clip above -3 dB.
imagesc (t, f, log(S));        # display in log scale
set (gca, "ydir", "normal");   # put the 'y' direction in the correct direction

%%% modulation d'amplitude de la 1ère sinusoïde
figure(2);
plot(t,S(541,:));
hold off;

%%% contenu fréquentiel pour la trame 10
figure(3);
plot(S(:,10));
hold off;
