# Eye-color sampling experiment

Eye color now reuses existing MediaPipe iris landmarks, samples lateral iris
sectors, excludes pupil and eyelids, then classifies HSV. Haar sampling remains
fallback when landmarks are missing. Confirmed eye score improved 1/3 to 2/3.

Hair segmentation was tested locally but did not improve the three confirmed
labels, so no new model was shipped.
