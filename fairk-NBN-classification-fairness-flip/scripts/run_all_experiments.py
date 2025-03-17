import pandas as pd

from scripts.visualize_script_v2 import visualize_dom_attr_vs_sen_attr, visualize_rprvsbpr, visualize_accuracy, \
    visualize_random, visualize_compare_rprvsbpr
from src.jobs.fairness_parity import FairnessParity
from src.utils.preprocess_utils import  flip_value


def get_accuracy(train_index, fp):
    accuracy_list = []

    test_stas_before = fp.get_test_statistics_df(fp.x_val, fp.y_val, fp.y_val_sensitive_attr)
    accuracy_list.append({"train_val_flipped": 0, "accuracy": test_stas_before["accuracy"]})
    for i, index in enumerate(train_index):
        fp.y_train = flip_value(fp.y_train, index, fp.class_positive_value, fp.class_attribute)
        fp._train()
        test_stas_before = fp.get_test_statistics_df(fp.x_val, fp.y_val, fp.y_val_sensitive_attr)

        accuracy_list.append({"train_val_flipped": i + 1, "accuracy": test_stas_before["accuracy"]})

    df = pd.DataFrame(accuracy_list)

    return df

experiment_runs =\
    [['color','red',"red_blue_dataset",'red_blue_'],
     ['Sex','female','diabetic_data_wtarget','diabetic_'],
     ['Sex','female','dutch_cencus_wtarget','dutch_cencus_'],
     ['Sex','female','law_dataset_wtarget','law_'],
     ['Sex','female','uci_cc_wtarget','uci_cc_'],
     ['Sex','female','german_credit_wtarget','german_credit_'],
     ['marital','married','bank_wtarget','bank_']]


ida = [True,True,False,'ida']
eda = [False,False,False,'eda']
data_list = []
for run in experiment_runs:
    print(run[3])

    fp = FairnessParity(
        knn_neighbors=1,
        class_attribute="class",
        sensitive_attribute=run[0],
        sensitive_attribute_protected=run[1],
        positive_class_value=1,
        split_percent = 0.1 if run[3] == 'law_' else 0.2,
        basic_split=True,
        random_train_point=False,
        exclude_sensitive_attribute=True,
        second_weight=ida[1],
        sensitive_catches_dominant=ida[2],
        load_from="../data/"+run[2]+".csv",
        experiment_name= run[3]+ida[3],
        local_dir_res="results/",
        local_dir_plt="plots/",
        csv_to_word=True
    )
    reslts_df_ida, train_indexer,test_stats = fp.run_fairness_parity()

    test_stats = test_stats.pivot(index="Metric", columns="State", values="Value")
    percentage_of_flips_train = (len(train_indexer) / len(fp.x_train)) * 100
    percentage_of_flips_t0 = (len(train_indexer) / len(fp.reverse_index_innit)) * 100

    data_list.append({
        "run": run[3],
        "rpr_before": test_stats.loc["sen_attr_positive_ratio", "Before"],
        "rpr_after": test_stats.loc["sen_attr_positive_ratio", "After"],
        "bpr_before": test_stats.loc["dom_attr_positive_ratio", "Before"],
        "bpr_after": test_stats.loc["dom_attr_positive_ratio", "After"],
        "ac_before": test_stats.loc["accuracy", "Before"],
        "ac_after": test_stats.loc["accuracy", "After"],
        "percentage_of_flips": percentage_of_flips_train,
        "percentage_of_flips_t0": percentage_of_flips_t0,
    })
    #ida_acc_df = get_accuracy(train_indexer,fp)
    #data_list.append({"Name": run[3]+ida[3], "Percentage": percentage_of_flips})

    fp = FairnessParity(
        knn_neighbors=1,
        class_attribute="class",
        sensitive_attribute=run[0],
        sensitive_attribute_protected=run[1],
        split_percent=0.1 if run[3] == 'law_' else 0.2,
        basic_split=True,
        random_train_point=False,
        positive_class_value=1,
        second_weight=eda[1],
        sensitive_catches_dominant=eda[2],
        load_from="../data/" + run[2] + ".csv",
        experiment_name=run[3] + eda[3],
        local_dir_res="results/",
        local_dir_plt="plots/",
        csv_to_word=True
    )
    reslts_df_eda, train_indexer,_ = fp.run_fairness_parity()

    fp = FairnessParity(
        knn_neighbors=1,
        class_attribute="class",
        sensitive_attribute=run[0],
        sensitive_attribute_protected=run[1],
        split_percent=0.1 if run[3] == 'law_' else 0.2,
        basic_split=True,
        random_train_point=True,
        positive_class_value=1,
        second_weight=eda[1],
        sensitive_catches_dominant=eda[2],
        load_from="../data/" + run[2] + ".csv",
        experiment_name=run[3] + eda[3],
        local_dir_res="results/",
        local_dir_plt="plots/",
        csv_to_word=True
    )
    #reslts_df_random, train_indexer, _ = fp.run_fairness_parity()
    #visualize_dom_attr_vs_sen_attr(reslts_df_ida,reslts_df_eda, run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_dom_attrVSsen_attr')

    #eda_acc_df = get_accuracy(train_indexer, fp)
    #visualize_random(reslts_df_ida,run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_random_rprVSBPR')
    #visualize_compare_rprvsbpr(reslts_df_ida,reslts_df_eda,reslts_df_random,run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_comparison_rprVSBPR')
    visualize_rprvsbpr(reslts_df_ida,reslts_df_eda, run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_rprVSBPR.png')
    #visualize_accuracy(ida_acc_df,eda_acc_df,run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_accuracy')
df = pd.DataFrame(data_list)
df.to_csv("val_data_beforeAndAfter.csv", index=False)


