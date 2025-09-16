# analyze_volume.R
# install DRAP from GitHub and lme4 if needed
# devtools::install_github('SCBIT-YYLab/DRAP')
library(DRAP)
library(lme4)
library(ggplot2)
df <- read.csv('data/tumor_volumes_mock.csv')
# example: fit linear mixed model on log(volume) with (1|Model)
df$logVol <- log(df$Volume_mm3 + 1)
mod <- lmer(logVol ~ Arm * Day + (1 | Model), data=df)
summary(mod)
# use DRAP functions to compute TGI or response labels if you have those conventions

