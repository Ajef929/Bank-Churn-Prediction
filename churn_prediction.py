import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import random
import copy
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids,RandomUnderSampler
from imblearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from statistics import Counter
import umap.umap_ as umap
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from seaborn import pairplot,heatmap,boxplot,violinplot

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

def initial_inspection(df) -> None:
    """printing the first rows and the summary statistics of the dataframe"""
    print("Dataframe head")
    print(df.head())

    print("Descriptive Statistics")
    print(df.describe())

def plot_histogram_single(df,selected_cols = [],save=False) -> None:
    """plotting a single histogram. Not completed"""
    print()



def plot_bar_charts(df,scale = False,range=3,target_col=None,selected_cols = [],save=False,save_title=None) -> None:
    """Plots bar charts for all or selected variables"""
    df_copy = copy.deepcopy(df) ##create a copy of the dataframe
    ##selecting only numeric columns apart from target_col
    numeric = df_copy.select_dtypes(include=np.number)

    if scale: ##scaling data beforehand
        scaler = MinMaxScaler()
        scaler.fit(numeric)
        scaled = scaler.transform(numeric)
        numeric = pd.DataFrame(scaled, columns=numeric.columns)
        if target_col is not None: numeric[target_col] = df_copy[target_col]
        df_copy = numeric
    
    if selected_cols == []:
        print("No Columns specified, plotting all numeric...")
    else: 
        df_copy = df_copy.loc[:,selected_cols] ##take a subset of the columns
    #boxplot(data = df_copy, orient="h")
    boxplot(data=df_copy, hue=target_col,whis=range,orient='h')
    plt.xticks(rotation=45) 

    if target_col is not None and scale:
        plt_title = f"Boxplot with {target_col} as hue and IQR multiplier of {range} scaled to values between 0 and 1"
    else:
        plt_title = f"Boxplot with IQR multiplier of {range}"
    plt.title(plt_title)
    if save: plt.savefig(f"Boxplot{save_title}")
    plt.show()

def plot_violin_charts(df,scale = False,target_col=None,selected_cols = [],save=False,save_title=None) -> None:
    """Plots violin charts for all or selected variables"""
    df_copy = copy.deepcopy(df) ##create a copy of the dataframe
    if scale:
        scaler = MinMaxScaler()
        scaler.fit(df_copy)
        scaled = scaler.transform(df_copy)
        df_copy = pd.DataFrame(scaled, columns=df_copy.columns)    
        
    if selected_cols == []:
        print("No Columns specified, plotting all numeric...")
    else: 
        df_copy = df_copy.loc[:,selected_cols] ##take a subset of the columns
    #boxplot(data = df_copy, orient="h")
    violinplot(data=df_copy, hue=target_col)
    if target_col is not None:
        plt_title = f"Violinplot with {target_col} as hue"
    else:
        plt_title = f"Violinplot"
    
    plt.title(plt_title)
    if save: plt.savefig(f"Boxplot{save_title}")
    plt.show()




def plot_scatter_matrix(df,selected_cols = [],target_col=None,pairplot_colour=None,save=False,save_title=None) -> None:
    """plot a matrix of scatter plots for all or selected varriables"""

    if selected_cols == []:
        print("No Columns specified, plotting all numeric...")
        pairplot(df,hue=pairplot_colour)
        plt.title("Pair Plot for all columns")
        if save: plt.savefig(f"ScatterplotMatrix{save_title}")
        plt.show()

    else:
        if len(selected_cols) == 1:
            print("plotting single column ...")
            plt.scatter(x=df.index, y=df[selected_cols[0]])
            
            if target_col is not None: 

                plt.title(f"Scatter plot of {selected_cols[0]} against {target_col}")
                plt.xlabel(f"{target_col}") ##plot ag
            else:
                plt.title(f"Scatter plot of {selected_cols[0]} against Index")
                plt.xlabel(f"Index") ##plot ag

            plt.ylabel(selected_cols[0])
            if save: plt.savefig(f"ScatterplotMatrix{save_title}")
            plt.show()
        ##if there are selected columns then take a subset of data that contains only those columns 
        else:
            df_subset = df.loc[:,selected_cols]
            pairplot(df_subset,hue=pairplot_colour)
            plt.title("Pair Plot for selected columns")
            if save: plt.savefig(f"ScatterplotMatrix{save_title}")
            plt.show()





def correlation_testing(df,selected_cols=[],method="spearman",save=False,isolate_strong_thresh=None,save_title=None) -> None:
    """plot the correlations of either all columns in the dataframe or selected columns"""
    df_copy = copy.deepcopy(df) ##create a copy of the dataframe

    if selected_cols == []:
        print("No Columns specified, plotting all numeric...")
    else: 
        df_copy = df_copy.loc[:,selected_cols] ##take a subset of the columns

    matrix = df_copy.corr(method=method).round(2) ##rounding for readability

    #if isolate_strong_thresh is not None: ##showing only strong correlations
        #matrix = matrix.unstack()
        #matrix = matrix[abs(matrix) >= isolate_strong_thresh]
        #matrix = matrix.stack()

    #mask = np.triu(np.ones_like(matrix, dtype=bool)) ##plotting only the triangular matrix 
    
    heatmap(matrix, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
    plt.title("Correlation Heatmap")
    if save: plt.savefig('CorrelationHeatmap{save_title}')
    plt.show()

def model_diagnosic_plots(model):
    """check the diagnostics of the model in question"""

    return 


def check_class_imbalance(df,target_class = "target",visualise=False,save=False,save_title=None) -> None:
    """function to check the class imbalance of the target variable"""
    values = df[target_class]
    c_dict = Counter(values) ##initialise
    ##binary class
    if len(values.unique()) == 2:
        print("Target class is binary")

    if len(values.unique()) > 2:
        print("Target class is multiclass")
    
    p_dict = {}
    total_values = sum(c_dict.values())
    for key,value in c_dict.items():
        print(f"Count for class {key}: {value}")
        ##creating a dictionary of proportions
        prop = value / total_values
        
        print(f"Proportion for class {key}: {prop}")
        p_dict[key] = prop
    
    if visualise:
        plt.bar(x=p_dict.keys(),height=p_dict.values())
        plt.title(f"Class Balance for {target_class}")
        plt.xlabel(target_class)
        plt.ylabel("Proportion")
        if save: plt.savefig(f"ClassImbalance{save_title}")
        plt.show()

    ##multiclass

def preprocess(df,dummy_cols=["diagnosis"],target_col = "Exited",id_cols=["ID"]) -> None:
    """Initial prepreprocessing for the data in quesion"""
    #convert all desired columns to dummy variables
    temp = pd.get_dummies(df,columns=dummy_cols)
    ##remove id variable(s)
    id_cols.append(target_col)
    X = temp.drop(id_cols,axis=1) ##ignoring id varible(s) and target column
    
    #column_names = list(X.columns) ##extrating column names
    y = np.array(pd.DataFrame(df.loc[:,target_col],columns=[target_col])) ##setting response variable
    return X,y

##performing some visualisation
#initial_inspection(df)

##checking for class imbalance
#check_class_imbalance(df,target_class = "Exited",visualise=True)

##modify pipeline between problems
def create_custom_pipeline() -> Pipeline:
    """creating a custom pipeline of models and preprocessing"""
    p = make_pipeline(
        RandomUnderSampler(random_state=0),
        BorderlineSMOTE(),
        #ClusterCentroids(random_state=0),
        Normalizer(),
        #umap.UMAP(),
        LogisticRegression(penalty='none',C=0.3,class_weight=None,solver='saga',max_iter=1000)
        )

    return p
    
def tuning_crossvalidation(pipe,X_train,y_train,search_implementation="Grid"):
    """implementation of tuning and cross validation with either Grid or Random search implementation"""
        
    if search_implementation == "Random":
        unif_comp = [round(np.random.uniform(2,25)) for i in range(100)]

        CV_mod =  RandomizedSearchCV(estimator= pipe,
                            param_distributions=params,
                            scoring=scoring_metric,
                            n_iter = n_iterations,
                            cv=cross_validation_folds,
                            random_state=random_seed,
                            refit='AUC',
                            return_train_score=False,
                            verbose=verbosity,
                            n_jobs=n_jobs
        )

    elif search_implementation =='Grid':

        CV_mod =  GridSearchCV(estimator= pipe,
                            param_grid=params,
                            scoring=scoring_metric,
                            cv=cross_validation_folds,
                            refit='AUC',
                            return_train_score=False,
                            verbose=verbosity,
                            n_jobs=n_jobs
        )

    CV_mod.fit(X_train,y_train)
    results = CV_mod.cv_results_
    best_estimator = CV_mod.best_estimator_
    return results,best_estimator

def print_results(results):
    """outputting means and standard deviations of cross validation"""
    means = results['mean_test_score']
    stds  = results['std_test_score']
    for mean, std, params in zip(means, stds, results['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

def plot_cv_training(results,param,figsize=(10, 10),ax_lims = (0.0,1.0),metric="AUC",save=False,save_title=None):
    """plotting the optimisation process of the tuning parameter specified"""
    plt.figure(figsize=figsize)
    plt.title(f" evaluating {param}",
              fontsize=16)
    plt.xlabel(f"{param}")
    plt.ylabel(metric)
    ax = plt.gca()
    # adjust these according to your accuracy results and range values.
    #ax.set_xlim(0, 700)
    lwr_lim,upr_lim = ax_lims
    ax.set_ylim(lwr_lim, upr_lim)
    # Get the regular numpy array from the MaskedArray
    try:
        X_axis = np.array(results[param].data, dtype=float)
    except ValueError:
        X_axis = np.array(results[param].data, dtype=str)

    scoring = ['roc_auc']
    
    sample_score_mean = results['mean_test_score']
    sample_score_std = results['std_test_score']
    
    ax.fill_between(X_axis, sample_score_mean - sample_score_std,sample_score_mean + sample_score_std)
    ax.plot(X_axis, sample_score_mean)


    best_index = np.nonzero(results['rank_test_score'] == 1)[0][0]
    best_score = results['mean_test_score'][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f with k=%s" % (best_score, X_axis[best_index]),
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    if save: plt.savefig(f"TrainingPlotParam{param}{save_title}")
    plt.show()


def evaluation(best_model,y_test,X_test,save=False,save_title=None):
    """Evaluating the performance of model using a ConfusionMatrix note that for preliminary evaluation can use X_train and y_train"""
    ##returning the best model
    y_true, y_pred = y_test, best_model.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("=============================\n")

    cm = confusion_matrix(y_test, y_pred)
    out = ConfusionMatrixDisplay(confusion_matrix=cm)
    out.plot()
    plt.title(f"Confustion Matrix for {save_title}")
    if save: plt.savefig(f'ConfusionMatrix{save_title}')
    plt.show()



def main():
    ##defining some parameters to work with
    split_amount = 0.1
    visualise_training = True #visualise cross validation training process
    use_sample = True
    target_col = "Exited"
    stratified_train_test_split = True

    df = pd.read_csv("Churn_Modelling.csv") 
    
    if use_sample:
        print("Sampling...")
        df = df.sample(n=10000)
        
    ##consider observation weighting
    ##consider other preprocessing
    initial_inspection(df)
    #check_class_imbalance(df,target_class = "Exited",visualise=True)
    #plot_scatter_matrix(df,selected_cols=["Age","Balance","CreditScore","EstimatedSalary","NumOfProducts","Tenure","Gender"],pairplot_colour="Gender",save=True,save_title="SelectedColumns")
    ##plotting all numeric
    #plot_scatter_matrix(df,save=True,save_title="AllNumeric")

    ##plotting selected variables against against response
    #plot_bar_charts(df,scale=True,save=True,save_title="Scaled Bar Charts")

    #selected_variables = ["Age","Balance","CreditScore","EstimatedSalary","NumOfProducts","Tenure"]
    #for var in selected_variables:
        #plot_scatter_matrix(df,selected_cols=[var],target_col=target_col,save=True,save_title=f"{target_col} against {var}")

    #plot_scatter_matrix(df,selected_cols=["EstimatedSalary","Tenure","Gender","Balance","CreditScore"],pairplot_colour="Gender",save=True,save_title="SelectedColumns")
    #correlation_testing(df,method="spearman",save=True,save_title="AllNumeric")
    ##stop code after visualisation
    ##some data preprocessing
    X,y = preprocess(df,dummy_cols=["Gender","Geography","Surname"], target_col=target_col,id_cols=["CustomerId","RowNumber"]) 

    #y = np.column_or_1d(y)
    stratify = None ##initialise strateficiation variable
    if stratified_train_test_split: stratify = y     ##ensuring train test split is stratefied when specified
    ##split data into training and test 
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=split_amount,stratify = y)
    ##create pipleline
    ##single model
    pipe = create_custom_pipeline()
    model = pipe.fit(X_train,y_train)
    print("Model Configuration")
    print(pipe) ##model configuaration
    ##evaluate on training data
    evaluation(model,y_test=y_train,X_test=X_train,save=True,save_title="LogisticRegressionTrainEval")
    #evaluate on test data
    evaluation(model,y_test=y_test,X_test=X_test,save=True,save_title="LogisticRegressionTestEval")

    ##cross validated model
    #declaring some global parameters
    global random_seed,n_iterations,verbosity,n_jobs,cross_validation_folds,scoring_metric,params
    
    search_implementation = "Grid" ##Defining the parameter tunign process. Alternatively "Random". Select to choose search method
    random_seed = 53
    n_iterations = 50 ##number of iterations for randomized search, if applicable.
    verbosity = 1
    n_jobs = 1
    cross_validation_folds = 10
    scoring_metric = "roc_auc"

    ##defining parameter grid/distribution
    params = {
    #'svc__C':[i/100 for i in range(10,100,10)],
    #'svc__gamma':["scale","auto"],
    #'umap__n_components':range(10,25,5),
    #'umap__n_neighbors':range(5,15,3)}
    'logisticregression__C': [i/100 for i in range(10,100,20)],
    'logisticregression__penalty':['l1', 'l2', 'elasticnet']
}
    ##performing cross validation 
    results,best_model = tuning_crossvalidation(pipe=pipe,X_train=X_train,y_train=y_train, search_implementation=search_implementation)
    
    print("="*20)
    print("Best Model Configuration")
    print(best_model)
    #print("="*20)
    #print("Score:",best_model.score(X=X_train))
    ##plotting the training process
    if visualise_training:  
        print("="*20)
        print(f"\nEvaluating {search_implementation} SearchCV\n")
        all_params = best_model.get_params()
        #print("All Parameters",all_params)
        for param in ['param_logisticregression__penalty',"param_logisticregression__C"]:
            plot_cv_training(results,param,save=True,save_title=f"Training for {param} parameter")

        print("="*20 )

    ##confusion_matrix evaluation
    ##evaluate best model on training data
    evaluation(model,y_test=y_train,X_test=X_train,save=True,save_title="LogisticRegressionBestCVTrainEval")
    #evaluate best model on test data
    evaluation(model,y_test=y_test,X_test=X_test,save=True,save_title="LogisticRegressionBestCVTestEval")


main()
