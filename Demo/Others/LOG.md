# Project

### Description

The project aims to generate landscape image based on the emotion and mood analysis of music. The brief progress is to first analyze the mood elicited from the music. Save the analyzed mood data into tokens recognized by a model. By training the model to connect mood data of music with image data, the model can generate corresponding emotional landscape photo.

### Steps

1. Starting from finding datasets. --> Useful_Resources[1 & 3]
2. Retrive music information from the music. Representative mood and emotional music information can be,
   - Tempo(Librosa frame tempo retrieval), Mode, Loudness/Dynamics(Librosa frame tempo retrieval), Melody, Rhythm... **_As many features as possible._**
3. Use, an example is Random Forest, to find the relation between extracted music information and VA values.
4. Predict VA values from a given song using the trained model.
5. Feed the VA values to train a image genration model (GAN). --> Useful_Resources[2]
6. Evaluate the model.

### Associated Emotion

| Structural Feature | Definition                                                                                                                  | Associated Emotions                                                                                              |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tempo              | The speed or pace of a musical piece                                                                                        | Fast tempo: excitement, anger. Slow tempo: sadness, serenity.                                                    |
| Mode               | The type of scale                                                                                                           | Major tonality: happiness, joy. Minor tonality: sadness.                                                         |
| Loudness(Dynamics) | The physical strength and amplitude of a sound                                                                              | Intensity, power, or anger                                                                                       |
| Melody             | The linear succession of musical tones that the listener perceives as a single entity. Sequence of notes, e.g., C, A, B,... | Complementing harmonies: happiness, relaxation, serenity. Clashing harmonies: excitement, anger, unpleasantness. |
| Rhythm             | The regularly recurring pattern or beat of a song                                                                           | Smooth/consistent rhythm: happiness, peace. Rough/irregular rhythm: amusement, uneasiness. Varied rhythm: joy.   |

### Useful Resources

1. [Music Emotion Recognition: Toward new, robust standards in personalized and context-sensitive applications](https://github.com/juansgomez87/datasets_emotion?tab=readme-ov-file)
2. [Emotional Landscape Image Generation Using Generative Adversarial Networks](https://openaccess.thecvf.com/content/ACCV2020/papers/Park_Emotional_Landscape_Image_Generation_Using_Generative_Adversarial_Networks_ACCV_2020_paper.pdf)
3. [Image-Emotion (Arousal and Valence) CGnA10766 Dataset](https://figshare.com/articles/dataset/CGnA10766_Dataset/5383105)
4. [PMEmo: A Dataset For Music Emotion Computing](https://github.com/HuiZhangDB/PMEmo?tab=readme-ov-file)

### Challenges

1. **_I can find the dataset from the above linl. However, the dataset already provide the emotions of songs. They are either perceived by the audience, or induced by experiments. The songs are also already classified into emotions in 2D dimention (VA model, valence and arousal) or categories (discrete emotions). What can I do to show that I train a model to using the given song labels and songs?_**
   - Given song labels (dimentional or categorical) can be y_train and y_test. Songs are the X_train and X_test. Let's use dimentional data, to better align with specific emotions perceived from landscape photos.
2. **_Music and image are the two source of emotions. Thus, both of them have quantitive values for expressing the emotions. In music part, we know tempo, melody, and etc information can induce an emotion. In the image side, what is that?_**
   - Still thinking.
3. **_How to quantify the emotions?_**
   - I choose the VA model. V is valence. Valence represents the level of pleasure. The lower value of valence indicates a negative emotion, and the higher value indicates a positive emotion. Arousal is a level of excitement. The smaller the arousal value, the calmer the emotion. The larger the value, the more active the sensation. By aligning VA model values with music information based on the given emotion of the music, we can have a corresponding music to emotion standard.
4. **_Ok. My thought is as follows. Assume I found a dataset of songs, each song is already labelled with VA values. After I extracted the music information using librosa or whatever method from the songs in the dataset, I want to try to match the extracted music information with the VA values. For example, there is a song labelled with VA value of valence equals 3 and arousal equals 4 on the scale of 1 to 9 of 1 being the lowest and 9 being the highest. Then Va values of 3 and 4 can be regarded as quit low mood, maybe sad, on the perspective of me. Then, I extracted some music information from the song. Maybe the tempo is 40 bpm and the melody is C, Eb, and G with certain numbers(I am not that into music, correct me if I have wrong concepts). How do I align or match the VA values and those music informationt together to predict the VA value from a new song? Use random forest to find the connections and for prediction?_**
   - The truth, there is no need to limit the music information only on those informatoin that is regarded as the most influencial to the music emotions, such as only consider Tempo instead of other. It is better to have as many features as possible. Like, Tempo (BPM), Chroma Features (12-dimensional vector for pitch classes), MFCCs (20 coefficients capturing timbre), Spectral Contrast (differences in spectral peaks/valleys), Zero-Crossing Rate (noisiness/percussiveness), RMS Energy (loudness/dynamics), Key/Mode (major/minor inferred from chroma or tonal features).
   - Do as following steps. **Normalization**: Scale features (e.g., StandardScaler) if using models sensitive to input range. **Feature Flattening**: Convert multi-dimensional features (e.g., chroma, MFCCs) into a flat vector. Mean? Variance? **Mode Detection**: Use chroma features to infer major/minor mode (e.g., C vs. C minor). **Concatenate VA values**: Add into the dataframe.
   - Use, an example is **Random Forest**, to handles non-linear relationship between VA values and music information.
5. **_How can I evaluate the accuracy between generated images and input music?_**
   - I would like to conduct a simple user feedback. Simply ask people to choose VA values of the song and the generated image, and calculate the errors between the two VA values.

### Notes

1. For human nature, this process of thinking of an image representing the emotions we feel can be more natural than transforming an existing image to match the emotions we feel. [Emotional Landscape Image Generation Using
   Generative Adversarial Networks](https://openaccess.thecvf.com/content/ACCV2020/papers/Park_Emotional_Landscape_Image_Generation_Using_Generative_Adversarial_Networks_ACCV_2020_paper.pdf)
2. A common mistake when train a model using supervised learning is feeding the model with data feature names but forget to add feature names to the unseen data when using the trained model for prediction.
3. Version 1 training results: **_Mean Squared Error: 0.8317149672365658_** / **_R^2 Score: 0.4074172312061686_**
