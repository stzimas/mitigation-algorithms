from scripts.visualize_script_v2 import visualize_dom_attr_vs_sen_attr
from src.jobs.fairness_parity import FairnessParity


experiment_runs =\
    [['color','red',"red_blue_dataset",'red_blue_'],
     ['Sex','female','diabetic_data_wtarget','diabetic_'],
     ['Sex','female','dutch_cencus_wtarget','dutch_cencus_'],
     ['Sex','female','law_dataset_wtarget','law_'],
     ['Sex','female','uci_cc_wtarget','uci_cc_'],
     ['marital','married','bank_wtarget','bank_']]


ida = [True,True,True,'ida']
eda = [False,False,True,'eda']
for run in experiment_runs:
    print(run[3])

    fp = FairnessParity(
        knn_neighbors=3,
        class_attribute="class",
        sensitive_attribute=run[0],
        sensitive_attribute_protected=run[1],
        positive_class_value=1,
        split_percent = 0.1 if run[3] == 'law_' else 0.2,
        second_weight=ida[1],
        sensitive_catches_dominant=ida[2],
        load_from="../data/"+run[2]+".csv",
        experiment_name= run[3]+ida[3],
        local_dir_res="results/",
        local_dir_plt="plots/",
        csv_to_word=True
    )

    reslts_df, train_indexer = fp.run_fairness_parity()
    visualize_dom_attr_vs_sen_attr(reslts_df, run[3] + ida[3] + '/plots/' + run[3] + ida[3] + '_dom_attrVSsen_attr')

    fp = FairnessParity(
        knn_neighbors=3,
        class_attribute="class",
        sensitive_attribute=run[0],
        sensitive_attribute_protected=run[1],
        split_percent=0.1 if run[3] == 'law_' else 0.2,
        positive_class_value=1,
        second_weight=eda[1],
        sensitive_catches_dominant=eda[2],
        load_from="../data/" + run[2] + ".csv",
        experiment_name=run[3] + eda[3],
        local_dir_res="results/",
        local_dir_plt="plots/",
        csv_to_word=True
    )

    reslts_df, train_indexer = fp.run_fairness_parity()
    visualize_dom_attr_vs_sen_attr(reslts_df, run[3] + eda[3] + '/plots/' + run[3] + eda[3] + '_dom_attrVSsen_attr')
