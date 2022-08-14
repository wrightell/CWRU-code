load('MysteryImage.mat')
m = 1456;
n = 2592;

x_data = cols/n;
y_data = (m/n)*(1 - rows/n);

data = [x_data';y_data'];

R(1).split = [];
R(1).dir = [];
R(1).data = data;
R(1).left = [];
R(1).right = [];
R(1).x = [0;1];
R(1).y = [.246;m/n];
R(1).color = mean(vals,1);
R(1).vals = vals;

leaves = (1);

figure(1)
tiledlayout(2,3,TileSpacing='none')
for i = 1:3000
    [R,leaves] = build_tree(R,leaves);
    if mod(i,500) == 0
        nexttile
        plot_tree(R,leaves)
        axis square
        axis off
        title(string(length(leaves)) + " leaves")
    end
end

figure(2)
plot_tree(R,leaves)
title(string(length(leaves)+" leaves"))
axis off



