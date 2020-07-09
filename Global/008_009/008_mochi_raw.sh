#!/bin/bash
#Export all enviroment variables
#$ -V
#Use current working directory
#$ -cwd
#Join stdout and stderr
#$ -j y
#Memory
#$ -l virtual_free=40G
#Time
#$ -q long-sl7
#$ -l h_rt=720:00:00
#$ -t 1-1
#$ -o /users/project/prj004631/pbaeza/2020_Global_Epistasis/044_mochi2/001_proteins/013_S7/qsub_out/
#$ -N s7_raw
#Parallel environment
#$ -pe smp 4

inputFile="/users/project/prj004631/pbaeza/2020_Global_Epistasis/044_mochi2/001_proteins/013_S7/data/mochi_table.txt"
outputPath="/users/project/prj004631/pbaeza/2020_Global_Epistasis/044_mochi2/001_proteins/013_S7/mochi_output/"
projectName="S7_RAW"
startStage="1"
stopStage="0"
numCores="4"
maxOrder="7"
significanceThreshold="0.05"
adjustmentMethod="bonferroni"

MoCHI --inputFile $inputFile --outputPath $outputPath --projectName $projectName --startStage $startStage --numCores $numCores --maxOrder $maxOrder --significanceThreshold $significanceThreshold --adjustmentMethod $adjustmentMethod
