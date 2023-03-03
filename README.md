# Generate Eurobeat lyrics with a word-level RNN model

**Author:** Eemil Ahonen, eeonaaah@student.jyu.fi

## Prerequisites

- Python 3.6+
- [Pandas 1.3.5](https://pandas.pydata.org/)
- [NumPy 1.19.5](https://numpy.org/)
- [Tensorflow 2.7.0](https://github.com/tensorflow/tensorflow)
- (Optional) [Wordcloud 1.5.0](https://github.com/amueller/word_cloud) (for the word cloud)
- (Optional) [Beautiful soup 4.10.0](https://www.crummy.com/software/BeautifulSoup/) (for data scraping)


## Usage

### Use the provided dataset
Open the [4_eurobeat_rnn_word_final_predictions_and_generator.ipynb](./4_eurobeat_rnn_word_final_predictions_and_generator.ipynb) notebook 
and follow the instructions in the **Generate lyrics using the trained model** section.


### Scrape the dataset
**Consider the ethics of data scraping!**
[Read more on ethical web scraping](https://www.empiricaldata.org/dataladyblog/a-guide-to-ethical-web-scraping)

1. Run the notebook [scrape_eurobeat_lyric_urls.ipynb](./scrape_eurobeat_lyric_urls.ipynb) 
to find the lyric pages or use the provided URLs in the [lyric_urls.txt](./data/lyric_urls.txt) file.
2. Run the [scrape_eurobeat_lyrics.ipynb](./scrape_eurobeat_lyrics.ipynb) 
notebook to extract the lyrics from the lyric URLs.


## Dataset
Dataset comes from https://www.eurobeat-prime.com/


## Generated samples
Seed string Seed string from Other Side of Night — Odyssey Eurobeat (Not in the dataset):
"I found the rest of me in what I thought was fantasy\nWith the last of doubt I can freely shout "

### Temperature: 0.5
```
i wanna feel you baby i need you
baby baby take me to the top
i wanna stay with you
every night and day
i know you can be mine
i need you now
i want to be your girl
dont you feel me
dont let me down
love is your game
you
```


### Temperature: 1.0
```
the world
i see your fire were ready just keep so right
yeah playing the night
let the music fall in love
dont let me go
we gonna be happy side
drive you tonight
and i try to keep on your destination
you can discover baby baby love you
how can pass on now for
```


### Temperature: 1.5
```
up
lay your body working
strange amp passion
eternally cos heat bang oooh funny
ya change the heart im magic desire ill do you ground
golden baby games and getting ready
tell your hand could ? a shining lady
shout believe love turn fire dont closer now
ill keep in the groove running with your save
```
