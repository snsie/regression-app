import { ComponentType } from "react";
import Hook from "./Hook";
import Framing from "./Framing";
import TheLine from "./TheLine";
import Recipe from "./Recipe";
import Ladder from "./Ladder";
import Forest from "./Forest";
import Boosting from "./Boosting";
import Foundation from "./Foundation";
import CrossValidation from "./CrossValidation";
import Diagnostics from "./Diagnostics";
import Leakage from "./Leakage";
import Splits from "./Splits";
import Compare from "./Compare";
import Nested from "./Nested";
import Checklist from "./Checklist";

export interface Section {
  id: string;
  part: string;
  title: string;
  lede?: string;
  Component: ComponentType;
}

export const SECTIONS: Section[] = [
  { id: "hook", part: "Frame", title: "Same patient, new tool", lede: "An AI assistant can fit a state-of-the-art model in seconds. That changes which skill matters most: checking the answer.", Component: Hook },
  { id: "framing", part: "Frame", title: "Every regression answers one question", lede: "y = f(x) + ε. A number to predict equals a pattern we can learn plus everything we can't.", Component: Framing },
  { id: "line", part: "Frame", title: "The line and its residuals", lede: "Fit the line yourself. Least squares is just the one line with the smallest sum of squared misses.", Component: TheLine },
  { id: "recipe", part: "Frame", title: "Linear regression is a recipe", lede: "Start from a baseline value, then add a fixed amount for each unit of each predictor. You can check it with a calculator.", Component: Recipe },
  { id: "ladder", part: "Model ladder", title: "The model ladder", lede: "Each rung is a different answer to the model-family question: more flexible, harder to read. All of them get the same verification.", Component: Ladder },
  { id: "forest", part: "Model ladder", title: "Random forest: a crowd of trees", lede: "Many imperfect decision trees, each grown on a different resample of patients, vote on the answer.", Component: Forest },
  { id: "boosting", part: "Model ladder", title: "Gradient boosting: fix the last mistake", lede: "Small trees built one after another, each fit to what the trees before it still get wrong.", Component: Boosting },
  { id: "foundation", part: "Model ladder", title: "Tabular foundation models", lede: "Pretrained on millions of synthetic tables. Your training rows become its context, and it returns a full predictive distribution.", Component: Foundation },
  { id: "cv", part: "Verify", title: "Grade on patients it hasn't seen", lede: "Training scores flatter every model. Cross-validation grades each patient with a model that never saw them.", Component: CrossValidation },
  { id: "diagnostics", part: "Verify", title: "Is it right for each kind of patient?", lede: "A single score can hide problems. Look at held-out predictions, residuals, learning curves and subgroups.", Component: Diagnostics },
  { id: "leakage", part: "Verify", title: "The assistant's code ran. Was it right?", lede: "“Build the most accurate model to predict motor UPDRS and tell me how good it is.” No rules given.", Component: Leakage },
  { id: "splits", part: "Verify", title: "The question decides the split", lede: "Parkinson's telemonitoring: same models, three honest-looking ways to split, three very different answers.", Component: Splits },
  { id: "compare", part: "Compare fairly", title: "No model always wins", lede: "Same models, same folds, four real datasets. The best model depends on the data and on how you split it.", Component: Compare },
  { id: "nested", part: "Compare fairly", title: "Tuning without cheating", lede: "Try many settings and report the best, and you have graded the tuning on the same folds it used to choose.", Component: Nested },
  { id: "checklist", part: "Compare fairly", title: "Your verification checklist", lede: "A model always gives you an answer. Your job is to check whether it deserves your trust.", Component: Checklist },
];
