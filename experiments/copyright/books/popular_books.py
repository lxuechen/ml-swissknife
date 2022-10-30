"""
Create prompt datasets for popular books.
"""

import re

import fire
from nltk.tokenize.treebank import TreebankWordDetokenizer, TreebankWordTokenizer
import tqdm

from ml_swissknife import utils

# @formatter:off
# Maps book name to first chunk of each popular book. Used to search for start of book.
name2start = {
    "Stephen Covey - The 7 Habits of Highly Effective People":
        "In more than 25 years of working with people in business, university, and marriage and family settings, I have come in contact with many individuals who have achieved an incredible degree of outward success, but have found themselves struggling with an inner hunger, a deep need for personal congruency and effectiveness and for healthy, growing relationships with other people.",

    "Harry_Potter_and_the_Half-Blood_Prince":
        "It was nearing midnight and the Prime Minister was sitting alone in his office, reading a long memo that was slipping through his brain without leaving the slightest trace of meaning behind. He was waiting for a call from the president of a far-distant country, and between wondering when the wretched man would telephone, and trying to suppress unpleasant memories of what had been a very long, tiring and difficult week, there was not much space in his head for anything else. ",

    "Harry_Potter_and_the_Order_of_the_Phoenix":
        "The hottest day of the summer so far was drawing to a close and a drowsy silence lay over the large, square houses of Privet Drive. Cars that were usually gleaming stood dusty in their drives and lawns that were once emerald green lay parched and yellowing – for the use of hosepipes had been banned due to drought. Deprived of their usual car-washing and lawn-mowing pursuits, the inhabitants of Privet Drive had retreated into the shade of their cool houses, windows thrown wide in the hope of tempting in a non-existent breeze. The only person left outdoors was a teenage boy who was lying flat on his back in a flowerbed outside number four.",

    "Harry_Potter_and_the_Deathly_Hallows":
        "The two men appeared out of nowhere, a few yards apart in the narrow, moonlit lane. For a second they stood quite still, wands directed at each other's chests; then, recognising each other, they stowed their wands beneath their cloaks and started walking briskly in the same direction.",

    "Harry_Potter_and_the_Prisoner_of_Azkaban":
        "Harry Potter was a highly unusual boy in many ways. For one thing, he hated the summer holidays more than any other time of year. For another, he really wanted to do his homework, but was forced to do it in secret, in the dead of night. And he also happened to be a wizard.",

    "Harry_Potter_and_the_Philosophers_Stone":
        "Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal, thank you very much. They were the last people you'd expect to be involved in anything strange or mysterious, because they just didn't hold with such nonsense.",

    "Harry_Potter_and_the_Goblet_of_Fire":
        "The villagers of Little Hangleton still called it 'the Riddle House', even though it had been many years since the Riddle family had lived there. It stood on a hill overlooking the village, some of its windows boarded, tiles missing from its roof, and ivy spreading unchecked over its face. Once a fine-looking manor, and easily the largest and grandest building for miles around, the Riddle House was now damp, derelict and unoccupied.",

    "Harry_Potter_and_the_Chamber_of_Secrets":
        "Not for the first time, an argument had broken out over breakfast at number four, Privet Drive. Mr Vernon Dursley had been woken in the early hours of the morning by a loud, hooting noise from his nephew Harry's room.",

    "Flowers in the Attic - V":
        "It is so appropriate to color hope yellow, like that sun we seldom saw. And as I begin to copy from the old memorandum journals that I kept for so long, a title comes as if inspired. Open the Window and Stand in the Sunshine. Yet, I hesitate to name our story that. For I think of us more as flowers in the attic. Paper flowers. Born so brightly colored, and fading duller through all those long, grim, dreary, nightmarish days when we were held prisoners of hope, and kept captives by greed. But, we were never to color even one of our paper blossoms yellow.",
    
    "All the Light We Cannot See" :
        "At dusk they pour from the sky. They blow across the ramparts, turn cartwheels over rooftops, flutter into the ravines between houses. Entire streets swirl with them, flashing white against the cobbles. Urgent message to the inhabitants of this town, they say. Depart immediately to open country.",

    "Charlie and the Chocolate Factory - Roald Dahl":
        "These two very old people are the father and mother of Mr Bucket. Their names are Grandpa Joe and Grandma Josephine.",
    
    "Charlotte's Web - E":
        "WHERE'S Papa going with that ax?",

    "What_to_Expect_When_Youre_Expecting":
        "SO YOU'VE MADE THE DECISION TO start a family",

    "What Color Is Your Parachute 2012":
        "If we had such a thing as a national bumper-sticker for our cars, the bumper-sticker of the year would be: ",

    "Wild Swans":
        "Granddad says all the Milbourn women are extraordinary.",

    "The Catcher in the Rye - J":
        "IF YOU REALLY WANT TO HEAR about it, the first thing you'll probably want to know is where I was born, and what my lousy childhood was like, and how my parents were occupied and all before they had me, and all that David Copperfield kind of crap, but I don't feel like going into it, if you want to know the truth.",

    "The_Kite_Runner":
        "I became what I am today at the age of twelve, on a frigid overcast day in the winter of 1975. I remember the precise moment, crouching behind a crumbling mud wall, peeking into the alley near the frozen creek. That was a long time ago, but it's wrong what they say about the past, I've learned, about how you can bury it. Because the past claws its way out. Looking back now, I realize I have been peeking into that deserted alley for the last twenty-six years.",

    "The Outsiders - S":
        "WHEN I STEPPED out into the bright sunlight from the darkness of the movie house, I had only two things on my mind: Paul Newman and a ride home. I was wishing I looked like Paul Newman—he looks tough and I don't—but I guess my own looks aren't so bad. I have light-brown, almost-red hair and greenish-gray eyes. I wish they were more gray, because I hate most guys that have green eyes, but I have to be content with what I have. My hair is longer than a lot of boys wear theirs, squared off in back and long at the front and sides, but I am a greaser and most of my neighborhood rarely bothers to get a haircut. Besides, I look better with long hair.",

    "James and the Giant Peach-Dahl":
        "UNTIL HE WAS FOUR years old, James Henry Trotter had had a happy life. He lived peacefully with his mother and father in a beautiful house beside the sea. There were always plenty of other children for him to play with, and there was the sandy beach for him to run about on, and the ocean to paddle in. It was the perfect life for a small boy.",

    "Jonathan Livingston Seagull-Bach":
        "It was morning, and the new sun sparkled gold across the ripples of a gentle sea.",
}
# @formatter:on


def main(
    in_path="/Users/xuechenli/data/books-memorization/top_texts_filtered.json",
    out_dir="/Users/xuechenli/data/books-memorization/",
    completion_tokens=2000,
    prefix_lengths=(5, 10, 25, 50, 125, 250,)
):
    popular_books = utils.jload(in_path)

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    for prefix_length in prefix_lengths:
        data = dict()
        for key, start in tqdm.tqdm(name2start.items()):
            book = popular_books[key]

            try:
                search_result = re.search(pattern=start, string=book)
                start_index = search_result.start()
            except AttributeError as e:
                import pdb
                pdb.set_trace()
            book_no_front_matter = book[start_index:]

            completion = detokenizer.detokenize(
                tokenizer.tokenize(book_no_front_matter)[:completion_tokens]
            )
            prefix = detokenizer.detokenize(
                tokenizer.tokenize(book_no_front_matter)[:prefix_length]
            )
            data[prefix] = completion
        metadata = dict(pairs_size=len(name2start), sanitized_pairs_size=len(name2start))
        utils.jdump(
            dict(data=data, metadata=metadata),
            utils.join(out_dir, f'popular_books-prefix_length_{prefix_length}.json'),
        )


if __name__ == "__main__":
    fire.Fire(main)
