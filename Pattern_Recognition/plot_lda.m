function [] = plot_lda(Z,labels,offset,num_levels)
    
    figure(2)
    tiledlayout(1,2)
    nexttile
    t = (labels == 1);
    plot(Z(1,t),Z(2,t),'k.')
    xlabel Q1
    ylabel Q2
    title('LDA (offset = '+ string(offset) + ', levels = ' + string(num_levels)+')')
    hold on
    t = (labels == 2);
    plot(Z(1,t),Z(2,t),'b.')
    t = (labels == 3);
    plot(Z(1,t),Z(2,t),'g.')
    t = (labels == 4);
    plot(Z(1,t),Z(2,t),'r.')
    t = (labels == 5);
    plot(Z(1,t),Z(2,t),'c.')
    hold off
    axis square

    nexttile
    t = (labels == 1);
    plot(Z(2,t),Z(3,t),'k.')
    xlabel Q2
    ylabel Q3
    title('LDA (offset = '+ string(offset) + ', levels = ' + string(num_levels)+')')
    hold on
    t = (labels == 2);
    plot(Z(2,t),Z(3,t),'b.')
    t = (labels == 3);
    plot(Z(2,t),Z(3,t),'g.')
    t = (labels == 4);
    plot(Z(2,t),Z(3,t),'r.')
    t = (labels == 5);
    plot(Z(2,t),Z(3,t),'c.')
    axis square
end
   