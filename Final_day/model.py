"""
model.py — Emotion classifier with 30+ emotions.
Uses TF-IDF + Logistic Regression trained on a large synthetic + GoEmotions-style dataset.
Run `python model.py` once to train and save the model, then start the API.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

MODEL_PATH = "emotion_model.pkl"

EMOTIONS = [
    "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "love", "optimism", "pessimism", "trust", "anticipation",
    "anxiety", "excitement", "boredom", "confusion", "relief",
    "pride", "shame", "guilt", "envy", "jealousy",
    "gratitude", "loneliness", "hope", "disappointment", "frustration",
    "amusement", "admiration", "grief", "nostalgia", "nervousness",
    "curiosity", "contentment", "awe", "embarrassment"
]

TRAINING_DATA = [
    # joy
    ("I am so happy today!", "joy"),
    ("This is the best day of my life!", "joy"),
    ("I feel wonderful and full of energy", "joy"),
    ("Everything is going great!", "joy"),
    ("I'm bursting with happiness!", "joy"),
    ("Life is beautiful and joyful", "joy"),
    ("I feel so cheerful and bright", "joy"),
    ("Today is a fantastic day!", "joy"),
    ("I'm overjoyed with the results!", "joy"),
    ("Pure happiness fills my heart", "joy"),
    ("I love every moment of this", "joy"),
    ("What a delightful experience!", "joy"),
    ("I'm ecstatic about the news!", "joy"),
    ("Feeling on top of the world!", "joy"),
    ("This makes me incredibly happy", "joy"),

    # sadness
    ("I feel so sad and empty", "sadness"),
    ("Everything feels hopeless right now", "sadness"),
    ("I can't stop crying", "sadness"),
    ("My heart is broken", "sadness"),
    ("I feel deeply unhappy", "sadness"),
    ("Nothing brings me joy anymore", "sadness"),
    ("I'm filled with sorrow", "sadness"),
    ("The grief is overwhelming", "sadness"),
    ("I feel miserable and alone", "sadness"),
    ("Things will never get better", "sadness"),
    ("I'm so heartbroken right now", "sadness"),
    ("Life feels bleak and dark", "sadness"),
    ("I feel a deep sense of loss", "sadness"),
    ("My soul feels heavy with sadness", "sadness"),
    ("Tears won't stop falling", "sadness"),

    # anger
    ("I am absolutely furious!", "anger"),
    ("This makes my blood boil", "anger"),
    ("I can't believe how rude that was", "anger"),
    ("I'm so angry I could scream", "anger"),
    ("That person infuriates me", "anger"),
    ("This is completely unacceptable!", "anger"),
    ("I'm outraged by this behavior", "anger"),
    ("How dare they do that to me!", "anger"),
    ("I'm seething with rage", "anger"),
    ("This injustice makes me so mad", "anger"),
    ("I'm boiling with anger", "anger"),
    ("That was absolutely infuriating", "anger"),
    ("I cannot stand this anymore!", "anger"),
    ("My anger is through the roof", "anger"),
    ("I'm livid about what happened", "anger"),

    # fear
    ("I'm terrified of what might happen", "fear"),
    ("I feel so scared right now", "fear"),
    ("The darkness frightens me deeply", "fear"),
    ("I'm paralyzed with fear", "fear"),
    ("Something terrible is going to happen", "fear"),
    ("I'm afraid to go outside alone", "fear"),
    ("The thought of failing scares me", "fear"),
    ("I'm shaking with fright", "fear"),
    ("Fear grips my heart tightly", "fear"),
    ("I dread what comes next", "fear"),
    ("I feel a terrible sense of dread", "fear"),
    ("The unknown terrifies me", "fear"),
    ("I can't shake this feeling of terror", "fear"),
    ("I'm petrified and can't move", "fear"),
    ("My biggest fears are coming true", "fear"),

    # surprise
    ("I can't believe that just happened!", "surprise"),
    ("Wow, I never expected this!", "surprise"),
    ("That completely shocked me!", "surprise"),
    ("This is totally unexpected!", "surprise"),
    ("I'm astonished by the news", "surprise"),
    ("No way, I'm completely stunned!", "surprise"),
    ("What a surprising turn of events!", "surprise"),
    ("I was caught completely off guard", "surprise"),
    ("That blew my mind!", "surprise"),
    ("I'm absolutely amazed!", "surprise"),
    ("That was totally out of the blue", "surprise"),
    ("I didn't see that coming at all", "surprise"),
    ("What a shocking revelation!", "surprise"),
    ("I'm flabbergasted by this news", "surprise"),
    ("This has left me utterly speechless", "surprise"),

    # disgust
    ("That is absolutely revolting", "disgust"),
    ("I feel sick looking at this", "disgust"),
    ("This disgusts me to the core", "disgust"),
    ("How gross and repulsive!", "disgust"),
    ("That behavior is reprehensible", "disgust"),
    ("I'm nauseated by this", "disgust"),
    ("This makes me want to throw up", "disgust"),
    ("That's utterly disgusting behavior", "disgust"),
    ("I find this morally repugnant", "disgust"),
    ("The sight of it made me gag", "disgust"),
    ("I'm revolted by what I saw", "disgust"),
    ("This is deeply repulsive to me", "disgust"),
    ("That smell is absolutely horrible", "disgust"),
    ("I can't stand this filth", "disgust"),
    ("Everything about it disgusts me", "disgust"),

    # love
    ("I love you with all my heart", "love"),
    ("You mean everything to me", "love"),
    ("I adore spending time with you", "love"),
    ("My love for you is endless", "love"),
    ("You are the light of my life", "love"),
    ("I cherish every moment with you", "love"),
    ("You make my heart sing", "love"),
    ("I'm deeply in love with you", "love"),
    ("My heart belongs to you always", "love"),
    ("You are my everything and more", "love"),
    ("I love them so much it hurts", "love"),
    ("Being with you fills me with love", "love"),
    ("You are my greatest love", "love"),
    ("I fall more in love each day", "love"),
    ("This feeling of love is overwhelming", "love"),

    # optimism
    ("Things will definitely get better", "optimism"),
    ("I believe the future is bright", "optimism"),
    ("Everything will work out fine", "optimism"),
    ("I'm confident we'll succeed", "optimism"),
    ("Better days are just ahead", "optimism"),
    ("I have faith everything will improve", "optimism"),
    ("The best is yet to come", "optimism"),
    ("I'm sure tomorrow will be amazing", "optimism"),
    ("Positive outcomes await us", "optimism"),
    ("I always look on the bright side", "optimism"),
    ("Good things are coming our way", "optimism"),
    ("I maintain hope for a better future", "optimism"),
    ("We will overcome every obstacle", "optimism"),
    ("Success is just around the corner", "optimism"),
    ("I see only good things ahead", "optimism"),

    # pessimism
    ("Nothing ever works out for me", "pessimism"),
    ("I doubt anything will improve", "pessimism"),
    ("Things are only going to get worse", "pessimism"),
    ("There's no point in trying anymore", "pessimism"),
    ("I expect the worst every time", "pessimism"),
    ("This will fail just like always", "pessimism"),
    ("I have no faith in the future", "pessimism"),
    ("Everything always goes wrong", "pessimism"),
    ("I can't expect anything good", "pessimism"),
    ("Doom is inevitable for me", "pessimism"),
    ("Hope is just setting yourself up for failure", "pessimism"),
    ("I see no way things can improve", "pessimism"),
    ("Failure is all I can expect", "pessimism"),
    ("I've stopped expecting good things", "pessimism"),
    ("The worst case scenario always happens", "pessimism"),

    # trust
    ("I completely trust your judgment", "trust"),
    ("You have always been reliable and honest", "trust"),
    ("I know you'll do the right thing", "trust"),
    ("You never let me down", "trust"),
    ("I have full confidence in you", "trust"),
    ("I can always count on you", "trust"),
    ("Your honesty means the world to me", "trust"),
    ("I believe in you wholeheartedly", "trust"),
    ("You are the most trustworthy person I know", "trust"),
    ("I have no doubt about your integrity", "trust"),
    ("My faith in you is unshakeable", "trust"),
    ("You are someone I completely rely on", "trust"),
    ("I trust you with my deepest secrets", "trust"),
    ("Your word is good as gold to me", "trust"),
    ("I know you'll always be there", "trust"),

    # anticipation
    ("I can't wait for tomorrow!", "anticipation"),
    ("I'm so excited for what's coming", "anticipation"),
    ("I eagerly await the results", "anticipation"),
    ("The suspense is killing me", "anticipation"),
    ("I'm counting down the days", "anticipation"),
    ("I look forward to the upcoming event", "anticipation"),
    ("I'm buzzing with anticipation", "anticipation"),
    ("I can barely contain my excitement for what's next", "anticipation"),
    ("Every day I think about what's coming", "anticipation"),
    ("I'm eagerly awaiting the news", "anticipation"),
    ("The wait is both exciting and agonizing", "anticipation"),
    ("I keep thinking about what's ahead", "anticipation"),
    ("I can't stop thinking about the future", "anticipation"),
    ("My heart races thinking about what's coming", "anticipation"),
    ("I'm ready and waiting with great eagerness", "anticipation"),

    # anxiety
    ("I feel so nervous and on edge", "anxiety"),
    ("My stomach is in knots with worry", "anxiety"),
    ("I can't stop worrying about things", "anxiety"),
    ("My heart is racing with anxiety", "anxiety"),
    ("I feel panicked and overwhelmed", "anxiety"),
    ("Everything makes me feel anxious", "anxiety"),
    ("I overthink everything constantly", "anxiety"),
    ("I'm riddled with constant worry", "anxiety"),
    ("My anxiety is out of control", "anxiety"),
    ("I feel tense and apprehensive", "anxiety"),
    ("I'm worried sick about the outcome", "anxiety"),
    ("The anxiety is consuming me", "anxiety"),
    ("I can't calm my racing mind", "anxiety"),
    ("Every little thing makes me worry", "anxiety"),
    ("I feel overwhelming dread and unease", "anxiety"),

    # excitement
    ("I'm so incredibly excited!", "excitement"),
    ("This is thrilling and amazing!", "excitement"),
    ("I can barely contain my excitement", "excitement"),
    ("My heart is pumping with excitement", "excitement"),
    ("I'm buzzing with pure excitement", "excitement"),
    ("This is the most exciting thing ever!", "excitement"),
    ("I'm charged up with excitement!", "excitement"),
    ("I feel electric with excitement", "excitement"),
    ("Everything feels so energizing right now", "excitement"),
    ("I'm thrilled beyond words!", "excitement"),
    ("The excitement is absolutely contagious", "excitement"),
    ("I've never been so pumped up before", "excitement"),
    ("This thrill is unlike anything I've felt", "excitement"),
    ("I'm exhilarated beyond belief!", "excitement"),
    ("Pure excitement courses through my veins", "excitement"),

    # boredom
    ("I'm so incredibly bored right now", "boredom"),
    ("Nothing interests me at all", "boredom"),
    ("I feel completely unstimulated", "boredom"),
    ("Everything seems dull and lifeless", "boredom"),
    ("I have nothing to do and it's boring", "boredom"),
    ("I'm yawning constantly from boredom", "boredom"),
    ("Time just drags on endlessly", "boredom"),
    ("I feel totally unengaged and flat", "boredom"),
    ("There's absolutely nothing exciting happening", "boredom"),
    ("I'm mind-numbingly bored", "boredom"),
    ("Every day feels the same and boring", "boredom"),
    ("I lack any motivation or interest", "boredom"),
    ("This is tedious beyond belief", "boredom"),
    ("I feel like time stands completely still", "boredom"),
    ("Monotony is slowly killing me", "boredom"),

    # confusion
    ("I have no idea what's going on", "confusion"),
    ("Everything is so confusing to me", "confusion"),
    ("I don't understand any of this", "confusion"),
    ("I'm completely lost and confused", "confusion"),
    ("Nothing makes sense anymore", "confusion"),
    ("I'm bewildered by all of this", "confusion"),
    ("I can't make sense of the situation", "confusion"),
    ("I feel totally disoriented right now", "confusion"),
    ("What is even happening here?", "confusion"),
    ("I'm puzzled and don't know what to do", "confusion"),
    ("My mind is completely tangled in confusion", "confusion"),
    ("I'm utterly perplexed by everything", "confusion"),
    ("I thought I understood but I'm lost", "confusion"),
    ("Everything feels muddled in my head", "confusion"),
    ("I can't figure out what's happening at all", "confusion"),

    # relief
    ("I'm so relieved it's finally over", "relief"),
    ("What a huge weight off my shoulders", "relief"),
    ("I can finally breathe again", "relief"),
    ("Thank goodness everything worked out", "relief"),
    ("I feel such incredible relief right now", "relief"),
    ("The tension finally lifted completely", "relief"),
    ("I'm so glad the worst is behind us", "relief"),
    ("Relief washed over me like a wave", "relief"),
    ("Everything turned out better than expected", "relief"),
    ("I feel lighter now that it's resolved", "relief"),
    ("I was so worried but now I'm relieved", "relief"),
    ("The crisis has passed and I'm relieved", "relief"),
    ("A deep sigh of relief escapes me", "relief"),
    ("All that worry was for nothing", "relief"),
    ("I feel peaceful and relieved now", "relief"),

    # pride
    ("I'm so proud of what I accomplished", "pride"),
    ("This achievement fills me with pride", "pride"),
    ("I feel proud of my hard work", "pride"),
    ("I'm beaming with pride right now", "pride"),
    ("What I've achieved is remarkable", "pride"),
    ("I worked hard and I deserve this", "pride"),
    ("My accomplishment fills my heart with pride", "pride"),
    ("I'm incredibly proud of my progress", "pride"),
    ("This success makes me stand tall", "pride"),
    ("I have every right to feel proud", "pride"),
    ("I feel dignified and proud of myself", "pride"),
    ("Achieving this goal fills me with pride", "pride"),
    ("I'm proud of who I've become", "pride"),
    ("My dedication paid off beautifully", "pride"),
    ("I am immensely proud of this work", "pride"),

    # shame
    ("I feel deeply ashamed of what I did", "shame"),
    ("My actions were shameful and wrong", "shame"),
    ("I can't face anyone after this", "shame"),
    ("I'm filled with shame and regret", "shame"),
    ("I've brought disgrace on myself", "shame"),
    ("I want to hide from the world", "shame"),
    ("My behavior was embarrassing and shameful", "shame"),
    ("I feel humiliated by my own actions", "shame"),
    ("The shame is consuming me inside", "shame"),
    ("I wish I could take back what I did", "shame"),
    ("I'm too ashamed to even look up", "shame"),
    ("This shame weighs heavily on me", "shame"),
    ("I let myself and others down deeply", "shame"),
    ("I can't forgive myself for this", "shame"),
    ("Shame burns in my chest constantly", "shame"),

    # guilt
    ("I feel so guilty about what I did", "guilt"),
    ("I should never have done that to them", "guilt"),
    ("The guilt is eating me alive", "guilt"),
    ("I know I hurt someone and feel terrible", "guilt"),
    ("I can't stop feeling guilty about it", "guilt"),
    ("My conscience won't let me rest", "guilt"),
    ("I deeply regret hurting that person", "guilt"),
    ("I feel responsible for what went wrong", "guilt"),
    ("This guilt follows me everywhere I go", "guilt"),
    ("I owe them a real apology", "guilt"),
    ("I betrayed their trust and feel guilty", "guilt"),
    ("My guilt over this is overwhelming", "guilt"),
    ("I made a terrible mistake and I know it", "guilt"),
    ("I feel morally wrong for what I did", "guilt"),
    ("The guilt won't let me sleep at night", "guilt"),

    # envy
    ("I envy everything they have", "envy"),
    ("Why do they get everything and not me", "envy"),
    ("I want what they have so badly", "envy"),
    ("Seeing their success makes me envious", "envy"),
    ("I can't help but envy their lifestyle", "envy"),
    ("I wish I had their opportunities", "envy"),
    ("Their achievements make me feel envious", "envy"),
    ("I'm green with envy over their success", "envy"),
    ("Why does everything good happen to them?", "envy"),
    ("I secretly envy their happiness", "envy"),
    ("Their good fortune makes me envious", "envy"),
    ("I covet what they have deeply", "envy"),
    ("Envy stirs inside me when I see them", "envy"),
    ("I want their life more than my own", "envy"),
    ("Their advantages fill me with envy", "envy"),

    # jealousy
    ("I'm jealous of their relationship", "jealousy"),
    ("The thought of losing them makes me jealous", "jealousy"),
    ("I feel possessive and jealous of them", "jealousy"),
    ("Seeing them with others fills me with jealousy", "jealousy"),
    ("I'm consumed by jealousy over them", "jealousy"),
    ("Jealousy burns in my heart constantly", "jealousy"),
    ("I can't bear the thought of someone else having them", "jealousy"),
    ("I feel irrationally jealous and insecure", "jealousy"),
    ("Jealousy makes me act unlike myself", "jealousy"),
    ("I'm possessively jealous and I know it", "jealousy"),
    ("The jealousy is tearing me apart", "jealousy"),
    ("I hate feeling so jealous all the time", "jealousy"),
    ("Jealous thoughts occupy my every moment", "jealousy"),
    ("I'm consumed by jealousy I can't control", "jealousy"),
    ("This jealousy is making me miserable", "jealousy"),

    # gratitude
    ("I'm so grateful for everything you've done", "gratitude"),
    ("Thank you from the bottom of my heart", "gratitude"),
    ("I feel deeply thankful for your kindness", "gratitude"),
    ("Your generosity fills me with gratitude", "gratitude"),
    ("I appreciate everything you do for me", "gratitude"),
    ("I feel blessed and incredibly thankful", "gratitude"),
    ("Words can't express my deep gratitude", "gratitude"),
    ("I'm so thankful to have you in my life", "gratitude"),
    ("My heart overflows with gratitude", "gratitude"),
    ("I am truly grateful for your support", "gratitude"),
    ("Thank you for being there when I needed you", "gratitude"),
    ("Your help means more than you'll ever know", "gratitude"),
    ("I feel profoundly grateful every single day", "gratitude"),
    ("Gratitude fills my heart when I think of you", "gratitude"),
    ("I couldn't have done this without your help", "gratitude"),

    # loneliness
    ("I feel so utterly alone in this world", "loneliness"),
    ("No one understands me or cares", "loneliness"),
    ("I'm surrounded by people yet feel alone", "loneliness"),
    ("The loneliness is unbearable at times", "loneliness"),
    ("I have no one to talk to at all", "loneliness"),
    ("I feel invisible and completely forgotten", "loneliness"),
    ("Loneliness eats away at me daily", "loneliness"),
    ("I miss having someone to connect with", "loneliness"),
    ("I feel deeply isolated and disconnected", "loneliness"),
    ("No one notices when I'm not around", "loneliness"),
    ("The silence around me feels crushing", "loneliness"),
    ("I feel cut off from the rest of the world", "loneliness"),
    ("Loneliness shadows every moment I have", "loneliness"),
    ("I long for real human connection", "loneliness"),
    ("My loneliness feels endless and deep", "loneliness"),

    # hope
    ("I believe things will get better", "hope"),
    ("There's a light at the end of the tunnel", "hope"),
    ("I have hope for a brighter tomorrow", "hope"),
    ("Things will improve, I know it", "hope"),
    ("I hold onto hope even in darkness", "hope"),
    ("Hope keeps me going every day", "hope"),
    ("A better future is within our reach", "hope"),
    ("I maintain hope no matter how hard it gets", "hope"),
    ("Hope is the anchor that keeps me steady", "hope"),
    ("I refuse to give up hope", "hope"),
    ("Hope fills my heart even now", "hope"),
    ("I believe the best is yet to come", "hope"),
    ("Things will change for the better", "hope"),
    ("I see reasons to be hopeful today", "hope"),
    ("Hope burns in my heart like a flame", "hope"),

    # disappointment
    ("I'm so disappointed with the results", "disappointment"),
    ("This is not what I expected at all", "disappointment"),
    ("I feel let down and deflated", "disappointment"),
    ("My expectations were not met at all", "disappointment"),
    ("I'm deeply disappointed with them", "disappointment"),
    ("I had high hopes but was let down", "disappointment"),
    ("The outcome is so disappointing", "disappointment"),
    ("I feel a deep sense of disappointment", "disappointment"),
    ("I expected better from this situation", "disappointment"),
    ("Disappointment weighs heavily on me", "disappointment"),
    ("I worked hard only to be disappointed", "disappointment"),
    ("I feel failed by people I trusted", "disappointment"),
    ("My trust was betrayed and I'm disappointed", "disappointment"),
    ("The let down I feel is profound", "disappointment"),
    ("I'm bitterly disappointed by what happened", "disappointment"),

    # frustration
    ("This is so incredibly frustrating!", "frustration"),
    ("I can't get anything to work right", "frustration"),
    ("Why does nothing ever go as planned?", "frustration"),
    ("I'm at my wit's end with frustration", "frustration"),
    ("This is driving me absolutely crazy", "frustration"),
    ("I'm frustrated with all these obstacles", "frustration"),
    ("Nothing I do seems to work", "frustration"),
    ("The frustration is becoming unbearable", "frustration"),
    ("I keep hitting the same walls every time", "frustration"),
    ("I feel so frustrated I could give up", "frustration"),
    ("Every attempt ends in failure", "frustration"),
    ("This constant struggle is exhausting", "frustration"),
    ("My frustration level is at its peak", "frustration"),
    ("I'm exasperated by the lack of progress", "frustration"),
    ("No matter what I do, nothing works", "frustration"),

    # amusement
    ("That was absolutely hilarious!", "amusement"),
    ("I laughed so hard at that joke", "amusement"),
    ("That is genuinely the funniest thing ever", "amusement"),
    ("I can't stop laughing about it", "amusement"),
    ("That really tickled my funny bone", "amusement"),
    ("I'm in stitches from laughing so hard", "amusement"),
    ("How entertaining and amusing that was", "amusement"),
    ("I was thoroughly amused by the show", "amusement"),
    ("That joke was absolutely priceless", "amusement"),
    ("I find this hilariously funny", "amusement"),
    ("That had me rolling on the floor laughing", "amusement"),
    ("I'm still giggling about what happened", "amusement"),
    ("That was so funny I nearly cried laughing", "amusement"),
    ("The humor in this is killing me", "amusement"),
    ("I'm completely amused by this whole situation", "amusement"),

    # admiration
    ("I deeply admire your courage and strength", "admiration"),
    ("You are someone I truly look up to", "admiration"),
    ("Your talent is absolutely breathtaking", "admiration"),
    ("I have the utmost respect for you", "admiration"),
    ("You inspire me with everything you do", "admiration"),
    ("I admire your wisdom and dedication deeply", "admiration"),
    ("You are an incredible role model for all", "admiration"),
    ("I'm in awe of your incredible achievements", "admiration"),
    ("Your resilience is truly admirable", "admiration"),
    ("I hold you in the highest regard", "admiration"),
    ("Your work is truly exceptional", "admiration"),
    ("I find your dedication deeply inspiring", "admiration"),
    ("You have qualities I deeply admire", "admiration"),
    ("Your character earns my greatest admiration", "admiration"),
    ("I look at you with deep admiration", "admiration"),

    # grief
    ("I'm overcome with grief and loss", "grief"),
    ("The pain of losing them is unbearable", "grief"),
    ("Grief consumes every part of me", "grief"),
    ("I can't accept that they're gone", "grief"),
    ("My heart is shattered with grief", "grief"),
    ("I grieve deeply every single day", "grief"),
    ("The loss feels like a constant wound", "grief"),
    ("Grief makes every day feel heavy", "grief"),
    ("I miss them more than words can say", "grief"),
    ("The hole they left will never be filled", "grief"),
    ("My grief knows no boundaries or end", "grief"),
    ("I'm drowning in sorrow and deep grief", "grief"),
    ("Nothing prepares you for this kind of grief", "grief"),
    ("The grief is both immense and profound", "grief"),
    ("I weep with grief every single night", "grief"),

    # nostalgia
    ("I miss the good old days so much", "nostalgia"),
    ("Those memories fill me with nostalgia", "nostalgia"),
    ("I long for simpler times in my past", "nostalgia"),
    ("Nostalgia washes over me like a wave", "nostalgia"),
    ("I wish I could go back to that time", "nostalgia"),
    ("Those days were so much better and happier", "nostalgia"),
    ("I feel wistful thinking of my childhood", "nostalgia"),
    ("I yearn for the days that have passed", "nostalgia"),
    ("Remembering the past fills me with nostalgia", "nostalgia"),
    ("I miss how things used to be before", "nostalgia"),
    ("Those were truly the golden days", "nostalgia"),
    ("Nostalgia makes me smile and ache at once", "nostalgia"),
    ("I treasure the memories of those times", "nostalgia"),
    ("The past holds such warmth in my heart", "nostalgia"),
    ("Old songs bring back strong nostalgic feelings", "nostalgia"),

    # nervousness
    ("I feel so nervous before the big event", "nervousness"),
    ("My hands are shaking with nerves", "nervousness"),
    ("I'm a bundle of nerves right now", "nervousness"),
    ("Nervousness has me tied in knots", "nervousness"),
    ("I feel jittery and apprehensive today", "nervousness"),
    ("My stomach is flipping with nerves", "nervousness"),
    ("I'm nervous about the upcoming challenge", "nervousness"),
    ("My voice shakes when I'm this nervous", "nervousness"),
    ("The nervousness before performing is intense", "nervousness"),
    ("I can feel my heart racing with nerves", "nervousness"),
    ("Nervousness clouds my every thought", "nervousness"),
    ("I feel uneasy and edgy with nerves", "nervousness"),
    ("My palms sweat when I'm nervous like this", "nervousness"),
    ("I'm almost paralyzed with nervousness", "nervousness"),
    ("The nerves before the test are overwhelming", "nervousness"),

    # curiosity
    ("I'm so curious about how this works", "curiosity"),
    ("I want to learn more about everything", "curiosity"),
    ("My curiosity is endless and insatiable", "curiosity"),
    ("I need to understand this more deeply", "curiosity"),
    ("I wonder about the mysteries of the universe", "curiosity"),
    ("I'm fascinated and want to know more", "curiosity"),
    ("My mind is buzzing with curious questions", "curiosity"),
    ("I want to explore and discover everything", "curiosity"),
    ("Curiosity drives everything I do", "curiosity"),
    ("I'm intrigued and want to investigate", "curiosity"),
    ("I can't resist asking why and how", "curiosity"),
    ("The world never stops fascinating me", "curiosity"),
    ("I love exploring new and unknown things", "curiosity"),
    ("My curiosity leads me to great places", "curiosity"),
    ("I always want to know more about things", "curiosity"),

    # contentment
    ("I feel perfectly at peace with life", "contentment"),
    ("Everything is exactly as it should be", "contentment"),
    ("I'm content with what I have today", "contentment"),
    ("Life feels balanced and beautifully calm", "contentment"),
    ("I'm satisfied with how things are now", "contentment"),
    ("A warm sense of contentment fills me", "contentment"),
    ("I feel serene and deeply content", "contentment"),
    ("There's nowhere else I'd rather be", "contentment"),
    ("I'm at peace with myself and my life", "contentment"),
    ("Contentment settles over me like a blanket", "contentment"),
    ("I have everything I truly need", "contentment"),
    ("Life feels full and completely satisfying", "contentment"),
    ("A quiet joy and contentment fills me", "contentment"),
    ("I feel whole and genuinely fulfilled", "contentment"),
    ("All is well and I feel truly content", "contentment"),

    # awe
    ("I'm completely in awe of this sight", "awe"),
    ("This is absolutely breathtaking and magnificent", "awe"),
    ("I feel overwhelmed by the sheer beauty", "awe"),
    ("The grandeur of this fills me with awe", "awe"),
    ("I'm humbled and awed by nature's power", "awe"),
    ("This experience leaves me utterly speechless", "awe"),
    ("The universe fills me with profound awe", "awe"),
    ("I'm struck by the majesty of this moment", "awe"),
    ("What an extraordinary and awe-inspiring sight", "awe"),
    ("I feel small yet connected to something vast", "awe"),
    ("Awe washes over me like nothing before", "awe"),
    ("The beauty of this moment is staggering", "awe"),
    ("I'm completely overwhelmed by this wonder", "awe"),
    ("Awe and reverence fill every part of me", "awe"),
    ("This is beyond anything I could have imagined", "awe"),

    # embarrassment
    ("I'm so embarrassed right now", "embarrassment"),
    ("I want to disappear from embarrassment", "embarrassment"),
    ("My face is burning with embarrassment", "embarrassment"),
    ("I can't believe I said that in public", "embarrassment"),
    ("I feel mortified by what just happened", "embarrassment"),
    ("The embarrassment is making me flush red", "embarrassment"),
    ("I'm totally humiliated and embarrassed", "embarrassment"),
    ("I made a fool of myself in front of everyone", "embarrassment"),
    ("I wish the ground would swallow me up", "embarrassment"),
    ("This situation is deeply embarrassing for me", "embarrassment"),
    ("I cringe thinking about what I did", "embarrassment"),
    ("The embarrassment just won't go away", "embarrassment"),
    ("I keep replaying the embarrassing moment", "embarrassment"),
    ("I've never felt so embarrassed before", "embarrassment"),
    ("My embarrassment knows absolutely no bounds", "embarrassment"),
]


def preprocess(text):
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def augment_data(texts, labels):
    augmented_texts  = list(texts)
    augmented_labels = list(labels)
    keyword_phrases = {
        "joy":            ["so happy","feeling happy","i am happy","very happy","feeling great","wonderful day","full of joy","delighted today","i feel joyful","cheerful today"],
        "sadness":        ["feeling sad","i am sad","very sad","deeply sad","so unhappy","filled with sadness","sad today","i feel sad","feeling down","i feel unhappy"],
        "anger":          ["i am angry","so angry","feeling angry","very angry","furious right now","full of rage","i am furious","i feel angry","i am mad","feeling mad"],
        "fear":           ["i am scared","feeling scared","very afraid","so frightened","full of fear","terrified now","i am terrified","feeling fearful","i feel afraid","i am afraid"],
        "surprise":       ["i am surprised","so surprised","totally shocked","very unexpected","completely stunned","i am amazed","feeling surprised","i feel shocked","what a surprise","i am astonished"],
        "disgust":        ["feeling disgusted","so disgusting","that is revolting","i am disgusted","deeply disgusted","makes me sick","i feel disgusted","this is gross","i feel sick","that is gross"],
        "love":           ["i love you","feeling love","full of love","i adore you","deeply in love","i care for you","i love them","i feel love","loving feeling","i feel loved"],
        "optimism":       ["feeling optimistic","i am optimistic","very positive","bright future ahead","things will improve","staying positive","i feel optimistic","i am hopeful about future","positive outlook","feeling upbeat"],
        "pessimism":      ["feeling pessimistic","nothing will work","things are hopeless","i am pessimistic","expecting failure","no hope at all","i feel pessimistic","nothing ever works","all is lost","i give up"],
        "trust":          ["i trust you","feeling trusted","very trustworthy","i believe you","fully reliable","i have faith in you","i feel trusted","you are honest","i rely on you","trusting feeling"],
        "anticipation":   ["i am eager","so excited for","looking forward","can not wait","eagerly waiting","very eager now","i feel anticipation","i await eagerly","waiting with excitement","i am anticipating"],
        "anxiety":        ["feeling anxious","i am anxious","very worried","so nervous inside","full of anxiety","constant worry","i feel anxious","overwhelmed with worry","anxious feeling","i am so worried"],
        "excitement":     ["i am excited","feeling excited","so thrilled","very exciting","full of excitement","extremely thrilled","i feel excited","exciting feeling","i am so excited","feeling exhilarated"],
        "boredom":        ["feeling bored","i am bored","so bored today","very boring","nothing to do","completely bored","i feel bored","this is boring","bored to death","i feel dull"],
        "confusion":      ["feeling confused","i am confused","very confusing","so lost now","no idea what","completely confused","i feel confused","this makes no sense","i do not understand","confusing situation"],
        "relief":         ["feeling relieved","i am relieved","such a relief","finally over","huge relief now","so much relief","i feel relieved","thank goodness","what a relief","great relief"],
        "pride":          ["feeling proud","i am proud","so proud today","very proud of","filled with pride","immense pride","i feel proud","proud feeling","i am so proud","pride in myself"],
        "shame":          ["feeling ashamed","i am ashamed","so ashamed","full of shame","deeply ashamed","overcome with shame","i feel ashamed","i am full of shame","shameful feeling","i feel shame"],
        "guilt":          ["feeling guilty","i am guilty","so guilty now","full of guilt","deep guilt","overcome with guilt","i feel guilty","guilt feeling","i am full of guilt","guilty conscience"],
        "envy":           ["feeling envious","i am envious","so envious","i envy them","full of envy","deeply envious","i feel envy","envious feeling","i am jealous of success","i want what they have"],
        "jealousy":       ["feeling jealous","i am jealous","so jealous","full of jealousy","very jealous now","deeply jealous","i feel jealous","jealous feeling","overcome by jealousy","possessive jealousy"],
        "gratitude":      ["feeling grateful","i am grateful","so thankful","very grateful today","full of gratitude","deeply thankful","i feel grateful","thankful feeling","i am so thankful","grateful heart"],
        "loneliness":     ["feeling lonely","i am lonely","so lonely","very alone now","full of loneliness","deeply lonely","i feel lonely","lonely feeling","no one is there","i feel so alone"],
        "hope":           ["feeling hopeful","i am hopeful","so hopeful","full of hope","very hopeful now","hopeful today","i feel hopeful","i have hope","hope fills me","hopeful feeling"],
        "disappointment": ["feeling disappointed","i am disappointed","so disappointed","very let down","deeply disappointed","full of disappointment","i feel disappointed","disappointed feeling","this is disappointing","what a letdown"],
        "frustration":    ["feeling frustrated","i am frustrated","so frustrated","very frustrating","full of frustration","deeply frustrated","i feel frustrated","frustrated feeling","this is frustrating","nothing is working"],
        "amusement":      ["feeling amused","i am amused","so funny","very amusing","full of amusement","that is hilarious","i feel amused","amused feeling","this is funny","i am laughing"],
        "admiration":     ["feeling admired","i admire you","so admirable","full of admiration","deeply admiring","i look up to you","i feel admiration","admiring feeling","you are amazing","i respect you deeply"],
        "grief":          ["feeling grief","i am grieving","deep grief","full of grief","overwhelming grief","i am in grief","i feel grief","grief stricken","i grieve deeply","lost in grief"],
        "nostalgia":      ["feeling nostalgic","i am nostalgic","so nostalgic","miss the past","longing for past","nostalgic feelings","i feel nostalgic","nostalgic today","i miss old times","longing for old days"],
        "nervousness":    ["feeling nervous","i am nervous","so nervous","very nervous now","full of nerves","extremely nervous","i feel nervous","nervous feeling","nerves are high","i am on edge"],
        "curiosity":      ["feeling curious","i am curious","so curious","very curious now","full of curiosity","deeply curious","i feel curious","curious feeling","i want to know","i wonder about this"],
        "contentment":    ["feeling content","i am content","so content","very satisfied","full of contentment","deeply satisfied","i feel content","satisfied feeling","i am at peace","feeling peaceful"],
        "awe":            ["feeling awe","i am in awe","so amazing","full of awe","completely awed","deeply awed","i feel awe","breathtaking feeling","i am awestruck","overwhelming beauty"],
        "embarrassment":  ["feeling embarrassed","i am embarrassed","so embarrassed","very embarrassed","full of embarrassment","deeply embarrassed","i feel embarrassed","embarrassed feeling","my face is red","i feel humiliated"],
    }
    for emotion, phrases in keyword_phrases.items():
        for phrase in phrases:
            augmented_texts.append(phrase)
            augmented_labels.append(emotion)
    return augmented_texts, augmented_labels


def train_model():
    raw_texts  = [item[0] for item in TRAINING_DATA]
    raw_labels = [item[1] for item in TRAINING_DATA]

    texts  = [preprocess(t) for t in raw_texts]
    labels = list(raw_labels)

    texts, labels = augment_data(texts, labels)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=10.0,
            solver="saga",
            class_weight="balanced"
        ))
    ])

    pipeline.fit(texts, labels)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained on {len(texts)} samples across {len(set(labels))} emotions.")
    print(f"Saved to {MODEL_PATH}")
    return pipeline


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("No saved model found. Training now...")
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


_model = None

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict_emotions(text: str) -> dict:
    model = get_model()
    clean = preprocess(text)
    proba = model.predict_proba([clean])[0]
    classes = model.classes_

    emotion_scores = sorted(
        zip(classes, proba),
        key=lambda x: x[1],
        reverse=True
    )

    top3 = [
        {"emotion": e, "confidence": round(float(s), 4)}
        for e, s in emotion_scores[:3]
    ]

    all_emotions = [
        {"emotion": e, "confidence": round(float(s), 4)}
        for e, s in emotion_scores
    ]

    return {
        "input_text": text,
        "top_3_predictions": top3,
        "primary_emotion": top3[0]["emotion"],
        "primary_confidence": top3[0]["confidence"],
        "all_emotions": all_emotions,
        "total_emotions": len(all_emotions)
    }


if __name__ == "__main__":
    train_model()