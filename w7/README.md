# C6 - Team 6 - Week 7

[Final Presentation - w5-7](w7/FinalPresentation.pdf)

This repository is based on [CVMasterActionRecognitionSpotting](https://github.com/arturxe2/CVMasterActionRecognitionSpotting). The modified files for this week include implementations focused on *ball action spotting*, exploring various architectural strategies to improve temporal precision and feature representation.

### Modified / Added Files:

- *main_spotting.py*  
  Similar to the baseline but updated to *select the best model based on validation mAP* instead of validation loss. This improves alignment with the evaluation metric used in spotting tasks.

- *model_spotting_SGPMixer.py*  
  Integrates a new model architecture using the EDSGPMIXERLayers class from the official [T-DEED GitHub repository](https://github.com/arturxe2/T-DEED/blob/main/model/modules.py). This module incorporates *SGP (Scalable-Granularity Perception)* to enhance temporal discriminability across multiple time scales.

- *model_spotting_X3DEncDec.py*  
  Implements an *encoder-decoder architecture* using *X3D* as the temporal feature extractor (encoder), followed by a decoder module that upsamples the temporal resolution.

- *model_spotting_TCN_SGP.py*  
  Combines *X3D* for feature extraction with a hybrid *TCN + SGP* head. This setup processes features at multiple temporal granularities.
- *model_spotting_TPN.py*  
  Implements a *Temporal Pyramid Network* using *X3D* as the base feature extractor. Multiple TPN branches with different dilation rates are applied in parallel to capture short-, mid-, and long-range temporal patterns, and then fused for per-frame classification.

---

*Note*: Make sure all modified files are placed correctly in the project structure to maintain expected functionality and compatibility with training and inference scripts.
