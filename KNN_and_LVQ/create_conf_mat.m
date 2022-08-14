function mat = create_conf_mat(predicted,actual)
    TP = 0;
    FP = 0;
    TN = 0;
    FN = 0;
    for i = 1:length(predicted)
        if predicted(i) == 0 && actual(i) == 0
            TN = TN + 1;
        end
        if predicted(i) == 0 && actual(i) == 1
            FN = FN + 1;
        end
        if predicted(i) == 1 && actual(i) == 0
            FP = FP + 1;
        end
        if predicted(i) == 1 && actual(i) == 1
            TP = TP + 1;
        end
        mat = [TP,FP;FN,TN];
    end