library(tidyverse)
library(gridExtra)

library(ModelMetrics)

library(caret)

library(reshape2)
library(pROC)

library(effsize)
library(ScottKnottESD)

save.fig.dir = './output/figure/'

preprocess <- function(x, reverse){
  colnames(x) <- c("variable","value")
  tmp <- lapply(split(x, x$variable), function(df) {
    # Filter columns containing "value"
    df <- df[, grep("value", names(df))]
    # Rename columns by removing ".value"
    names(df) <- gsub(".value", "", names(df))
    return(df)
  })
  # Combine the list of data frames into a single data frame
  df <- do.call(cbind, tmp)
  
  # Compute ranking based on sk_esd function
  if(reverse == TRUE) { 
    ranking <- (max(sk_esd(df)$group) - sk_esd(df)$group) + 1 
  } else { 
    ranking <- sk_esd(df)$group 
  }
  
  # Add ranking as a new column to the original data frame
  x$rank <- paste("Rank", ranking[as.character(x$variable)])
  return(x)
}


get.file.level.metrics = function(df.file)
{
  all.gt = df.file$file.level.ground.truth
  all.prob = df.file$prediction.prob
  all.pred = df.file$prediction.label
  
  confusion.mat = caret::confusionMatrix(factor(all.pred), reference = factor(all.gt))
  # confusion.mat = confusionMatrix(all.pred, reference = all.gt)
  
  bal.acc = confusion.mat$byClass["Balanced Accuracy"]
  AUC = pROC::auc(all.gt, all.prob)
  
  # TP <- confusion.mat$table[2, 2]  # TP
  # TN <- confusion.mat$table[1, 1]  # TN
  # FP <- confusion.mat$table[1, 2]  # FP
  # FN <- confusion.mat$table[2, 1]  # FN


  # 计算平衡准确率（Balanced Accuracy）
  # TPR <- TP / (TP + FN)  # 真正率
  # TNR <- TN / (TN + FP)  # 真负率

  # levels(all.pred)[levels(all.pred)=="False"] = 0
  # levels(all.pred)[levels(all.pred)=="True"] = 1
  # levels(all.gt)[levels(all.gt)=="False"] = 0
  # levels(all.gt)[levels(all.gt)=="True"] = 1
  all.pred[all.pred=="False"] = 0
  all.pred[all.pred=="True"] = 1
  all.gt[all.gt=="False"] = 0
  all.gt[all.gt=="True"] = 1

  # all.gt = as.numeric_version(all.gt)
  all.gt = as.numeric(all.gt)
  
  # all.pred = as.numeric_version(all.pred)
  all.pred = as.numeric(all.pred)
  
  MCC = mcc(all.gt, all.pred, cutoff = 0.5)
  
  if(is.nan(MCC))
  {
    MCC = 0
  }
    
  # print(paste0("FN?: ", confusion.mat$table[3]))
  # print(paste0("TP?: ", confusion.mat$table[4]))
  # print(paste0("TN?: ", confusion.mat$table[1]))
  # print(paste0("FP?: ", confusion.mat$table[2]))

  TPR <- confusion.mat$table[4] / (confusion.mat$table[4] + confusion.mat$table[3])
  TNR <- confusion.mat$table[1] / (confusion.mat$table[2] + confusion.mat$table[1])
  
  eval.result = c(AUC, MCC, bal.acc, TPR, TNR)
  
  return(eval.result)
}

get.file.level.eval.result = function(prediction.dir, method.name)
{
  all_files = list.files(prediction.dir)
  
  all.auc = c()
  all.mcc = c()
  all.bal.acc = c()
  all.TPR <- c()
  all.TNR <- c()
  all.test.rels = c()
  
  for(f in all_files) # for looping through files
  {
    df = read.csv(paste0(prediction.dir, f))
    
    df = as_tibble(df)
    df = select(df, c(train, test, filename, file.level.ground.truth, prediction.prob, prediction.label))
    
    df = distinct(df)
    
    file.level.result = get.file.level.metrics(df)
    
    AUC = file.level.result[1]
    MCC = file.level.result[2]
    bal.acc = file.level.result[3]
    TPR <- file.level.result[4]
    TNR <- file.level.result[5]
    
    all.auc = append(all.auc,AUC)
    all.mcc = append(all.mcc,MCC)
    all.bal.acc = append(all.bal.acc,bal.acc)
    all.TPR <- append(all.TPR, TPR)
    all.TNR <- append(all.TNR, TNR)
    all.test.rels = append(all.test.rels,f)
    
  }
  
  result.df = data.frame(all.auc,all.mcc,all.bal.acc, all.TPR, all.TNR)
  
  
  all.test.rels = str_replace(all.test.rels, ".csv", "")
  
  result.df$release = all.test.rels
  result.df$technique = method.name
  
  return(result.df)
}

prediction_dir = '../../output/prediction/DeepLineDP/within-release/'
focal_dir = './focal_loss/output/prediction/DeepLineDP/within-release/'
dice_dir = './CB_loss/output/prediction/DeepLineDP/within-release/'
no_loss_dir = './no_loss/output/prediction/DeepLineDP/within-release/'

BCE.result = get.file.level.eval.result(prediction_dir, "Weighted_BCE")
no.result = get.file.level.eval.result(no_loss_dir, "Unweighted_BCE")
focal.result = get.file.level.eval.result(focal_dir, "Focal")
dice.result = get.file.level.eval.result(dice_dir, "CB")

all.result = rbind(BCE.result, no.result, focal.result, dice.result)

names(all.result) = c("AUC","MCC","Balance.Accuracy", "TPR", "TNR", "Release", "Technique")

auc.result = select(all.result, c("Technique","AUC"))
auc.result = preprocess(auc.result,FALSE)

mcc.result = select(all.result, c("Technique","MCC"))
mcc.result = preprocess(mcc.result,FALSE)

bal.acc.result = select(all.result, c("Technique","Balance.Accuracy"))
bal.acc.result = preprocess(bal.acc.result,FALSE)

TPR.result = select(all.result, c("Technique","TPR"))
TPR.result = preprocess(TPR.result,FALSE)

TNR.result = select(all.result, c("Technique","TNR"))
TNR.result = preprocess(TNR.result,FALSE)

p_auc <- ggplot(auc.result, aes(x=reorder(variable, -value, FUN=median), y=value)) + geom_boxplot() + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("AUC") + 
  xlab("") + 
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir,"file-AUC.png"), plot=p_auc, width=4,height=3)

p_bal_acc <- ggplot(bal.acc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() +
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("Balance Accuracy") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir,"file-Balance_Accuracy.png"), plot=p_bal_acc, width=4,height=3)

p_mcc <- ggplot(mcc.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("MCC") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-MCC.png"), plot=p_mcc, width=4,height=3)

p_TPR <- ggplot(TPR.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot() +
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("TPR") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir,"file-TPR.png"), plot=p_TPR, width=4,height=3)

p_TNR <- ggplot(TNR.result, aes(x=reorder(variable, value, FUN=median), y=value)) + geom_boxplot()  + 
  stat_summary(fun = median, geom = "text", vjust = -1, aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "blue") +
  stat_summary(fun = median, geom = "point", aes(label = sprintf("%.3f", after_stat(y))), size = 2, color = "red", shape = 18) +
  facet_grid(~rank, drop=TRUE, scales = "free", space = "free") + 
  ylab("TNR") + 
  xlab("") +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave(paste0(save.fig.dir, "file-TNR.png"), plot=p_TNR, width=4,height=3)


combined_plot <- grid.arrange(p_auc, p_bal_acc, p_mcc, p_TPR, p_TNR, ncol = 5)

# 保存合并后的图形
ggsave(paste0(save.fig.dir, "file_results.png"), plot = combined_plot, width = 20, height = 3)

## get within-project result
BCE.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")
no.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")
focal.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")
dice.result$project = c("activemq", "activemq", "activemq", "camel", "camel", "derby", "groovy", "hbase", "hive", "jruby", "jruby", "lucene", "lucene", "wicket")

BCE.file.level.by.project = BCE.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc), mean.TPR = mean(all.TPR), mean.TNR = mean(all.TNR))
no.file.level.by.project = no.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc), mean.TPR = mean(all.TPR), mean.TNR = mean(all.TNR))
focal.file.level.by.project = focal.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc), mean.TPR = mean(all.TPR), mean.TNR = mean(all.TNR))
dice.file.level.by.project = dice.result %>% group_by(project) %>% summarise(mean.AUC = mean(all.auc), mean.MCC = mean(all.mcc), mean.bal.acc = mean(all.bal.acc), mean.TPR = mean(all.TPR), mean.TNR = mean(all.TNR))

names(BCE.file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy", "TPR", "TNR")
names(no.file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy", "TPR", "TNR")
names(focal.file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy", "TPR", "TNR")
names(dice.file.level.by.project) = c("project", "AUC", "MCC", "Balance Accurracy", "TPR", "TNR")

# 将 file.level.by.project 写入 CSV 文件
write.csv(BCE.file.level.by.project, file = "BCE_file_level_by_project.csv", row.names = FALSE)
write.csv(no.file.level.by.project, file = "no_file_level_by_project.csv", row.names = FALSE)
write.csv(focal.file.level.by.project, file = "focal_file_level_by_project.csv", row.names = FALSE)
write.csv(dice.file.level.by.project, file = "CB_file_level_by_project.csv", row.names = FALSE)



