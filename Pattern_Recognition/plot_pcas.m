function [] = plot_pcas(PC4,labels,offset,num_levels)
    
    figure(1)
    tiledlayout(1,2)
    nexttile
    t = (labels == 1);
    plot(PC4(1,t),PC4(2,t),'k.')
    xlabel('PC1')
    ylabel PC2
    title('PCA (offset = '+ string(offset) + ', levels = ' + string(num_levels)+')')
    hold on
    t = (labels == 2);
    plot(PC4(1,t),PC4(2,t),'g.')
    t = (labels == 3);
    plot(PC4(1,t),PC4(2,t),'b.')
    t = (labels == 4);
    plot(PC4(1,t),PC4(2,t),'r.')
    t = (labels == 5);
    plot(PC4(1,t),PC4(2,t),'c.')
    axis square

    hold off
    nexttile
    t = (labels == 1);
    plot(PC4(2,t),PC4(3,t),'k.')
    xlabel('PC2')
    ylabel PC3
    title('PCA (offset = '+ string(offset) + ', levels = ' + string(num_levels)+')')
    hold on
    t = (labels == 2);
    plot(PC4(2,t),PC4(3,t),'g.')
    t = (labels == 3);
    plot(PC4(2,t),PC4(3,t),'b.')
    t = (labels == 4);
    plot(PC4(2,t),PC4(3,t),'r.')
    t = (labels == 5);
    plot(PC4(2,t),PC4(3,t),'c.')
    axis square
end
   