# ------------------------------------------------------------ #
# -------------- ADVERTISEMENT PROPERTIES INPUT -------------- #
# ------------------------------------------------------------ #

# Text
vBenefitsText = "Helps lower blood sugar in adults with type 2 diabetes."
vRisksText = "JANUVIA should not be used in patients with type 1 " \
             "diabetes or with diabetic ketoacidosis (increased ketones " \
             "in the blood urine). If you have had pancreatic (inflammation " \
             "of the pancreas), it is not known if you have a higher chance " \
             "of getting it while taking JANUVIA."

# Colors
vRGBBenefitsText = (53, 61, 126)
vRGBRisksText = (53, 61, 126)
vRGBBenefitsBackground = (193, 225, 223)
vRGBRisksBackground = (255, 255, 255)

# Visibility
vBenefitsVisibilityRatio = 1
vRisksVisibilityRatio = 0.15

# Average colors of image used and the remaining part from google.vision applet
vRGBImage = False
vRGBRest = (194, 211, 219)

# Social proof from images
vImagePersonExists = False

# ------------------------------------------------------------ #
# -------------------- BACK-END FUNCTIONS -------------------- #
# ------------------------------------------------------------ #

# Import libraries
import textstat  # Readability score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Tone
import re  # Regex
from nltk import word_tokenize, pos_tag, punkt # tenses

# Calculate final heart score
def CalculateHEARTScore(pRisksText,
                        pBenefitsText,
                        pRGBBenefitsText,
                        pRGBRisksText,
                        pRGBBenefitsBackground,
                        pRGBRisksBackground,
                        pBenefitsVisibilityRatio,
                        pRisksVisibilityRatio,
                        pRGBImage,
                        pRGBRest,
                        pImagePersonExists):

    print("\n")
    # Component I
    vReadabilityGap = MeasureReadabilityGap(pBenefitsText, pRisksText)
    print("Readability gap score (-1 to 1): " + str(vReadabilityGap))

    # Component II
    vColorSalienceGap = MeasureColorSalienceGap(pRGBBenefitsText, pRGBRisksText, pRGBBenefitsBackground, pRGBRisksBackground)
    print("Color salience gap score (-1 to 1): " + str(vColorSalienceGap))

    # Component III
    vToneGap = MeasureToneGap(pBenefitsText, pRisksText)
    print("Tone gap score (-1 to 1): " + str(vToneGap))

    # Component IV
    vTextQuantityGap = MeasureTextQuantityGap(pBenefitsText, pRisksText)
    print("Text quantity gap score (-1 to 1): " + str(vTextQuantityGap))

    # Component V
    vContentVisibilityGap = MeasureContentVisibilityGap(pBenefitsVisibilityRatio, pRisksVisibilityRatio)
    print("Content visibility gap score (-1 to 1): " + str(vContentVisibilityGap))

    # Component VI
    vImageSalienceGap = MeasureImageSalienceGap(pRGBImage, pRGBRest)
    print("Image salience gap score (-1 to 1): " + str(vImageSalienceGap))

    # Component VII
    vImageSocialProof = MeasureImageSocialProof(pImagePersonExists)
    print("Image social proof score (-1 to 1): " + str(vImageSocialProof))

    # Component VII
    vPresentBiasGap = MeasurePresentBiasGap(pBenefitsText, pRisksText)
    print("Present bias gap score (-1 to 1): " + str(vPresentBiasGap))

    # Final score
    vHEARTScore = round((vReadabilityGap +
                         vColorSalienceGap +
                         vToneGap +
                         vTextQuantityGap +
                         vContentVisibilityGap +
                         vImageSalienceGap +
                         vImageSocialProof +
                         vPresentBiasGap) /8, 3)
    print("\n")
    print("The HEART score (-1 to 1): " + str(vHEARTScore))

    print("The HEART score (percent): " + str((vHEARTScore + 1/0.02)) + "%")

# Measure the readability gap
def MeasureReadabilityGap(pBenefitsText, pRisksText):

    # Calculating the Flesch Reading-ease score gap
    vDifferenceFlesch = textstat.flesch_reading_ease(pBenefitsText) - textstat.flesch_reading_ease(pRisksText)

    # Standardize
    vDifferenceFleschStandard = vDifferenceFlesch/10

    # Calculating the Fog Scale score
    vDifferenceFog = (textstat.gunning_fog(pBenefitsText) if
                      textstat.gunning_fog(pBenefitsText) <= 12 else 12) - \
                     (textstat.gunning_fog(pRisksText) if textstat.gunning_fog(pRisksText) <= 12 else 12)

    # Standardize
    vDifferenceFogStandard = vDifferenceFog/12

    return round((vDifferenceFleschStandard + vDifferenceFogStandard)/2, 3)

# Measure the tone gap
def MeasureToneGap(pBenefitsText, pRisksText):

    # Sentiment analyzer tools
    vVaderToneAnalyser = SentimentIntensityAnalyzer()

    def SentimentAnalyzerScores(pDialogue):
        vScore = vVaderToneAnalyser.polarity_scores(pDialogue)
        return dict(vScore)

    vWeights = [1, 0, 1, 0]

    def GetAggregateTone(pText, pWeights):
        vAggregateTone = 0
        for x, y in zip(SentimentAnalyzerScores(pText).values(), pWeights):
            vAggregateTone = vAggregateTone + (x * y)
        return vAggregateTone

    return round(GetAggregateTone(pBenefitsText, vWeights) - GetAggregateTone(pRisksText, vWeights), 3)

# Measure the color salience gap
def MeasureColorSalienceGap(pRGBBenefitsText, pRGBRisksText, pRGBBenefitsBackground, pRGBRisksBackground):

    # Relative luminance of benefits
    if (CalculateLuminance(pRGBBenefitsText) <= CalculateLuminance(pRGBBenefitsBackground)):
        L1 = CalculateLuminance(pRGBBenefitsText)
        L2 = CalculateLuminance(pRGBBenefitsBackground)
    else:
        L1 = CalculateLuminance(pRGBBenefitsBackground)
        L2 = CalculateLuminance(pRGBBenefitsText)

    vRelLuminanceBenefits = (L1 + 0.05)/(L2 + 0.05)

    # Relative luminance of risks
    if (CalculateLuminance(pRGBRisksText) <= CalculateLuminance(pRGBRisksBackground)):
        L1 = CalculateLuminance(pRGBRisksText)
        L2 = CalculateLuminance(pRGBRisksBackground)
    else:
        L1 = CalculateLuminance(pRGBRisksBackground)
        L2 = CalculateLuminance(pRGBRisksText)

    vRelLuminanceRisks = (L1 + 0.05)/(L2 + 0.05)

    return round((vRelLuminanceBenefits - vRelLuminanceRisks)/40, 3)

# Measure the text quantity gap
def MeasureTextQuantityGap(pBenefitsText, pRisksText):

    # Functions
    def TextLength(pText):
        return len(pText)
    def TextWords(pText):
        return len(pText.split())
    def TextSentences(pText):
        return len(re.split(r'[.!?]+', pText))
    def StandardDifference(Function, pText1, pText2):
        return (Function(pText1) - Function(pText2))/max(Function(pText1), Function(pText2))

    return round((StandardDifference(TextLength, pBenefitsText, pRisksText) +
                  StandardDifference(TextWords, pBenefitsText, pRisksText) +
                  StandardDifference(TextSentences, pBenefitsText, pRisksText))/3, 3)

# Measure the visibility ratio gap
def MeasureContentVisibilityGap(pBenefitsVisibilityRatio, pRisksVisibilityRatio):
    return round(pBenefitsVisibilityRatio - pRisksVisibilityRatio, 3)

# Measure the color salience gap
def MeasureImageSalienceGap(pRGBImage, pRGBRest):

    if pRGBImage == False:
        return 1
    elif pRGBRest == False:
        return -1
    else:
        # Luminance of image
        L1 = CalculateLuminance(pRGBImage)

        # Luminance of the rest
        L2 = CalculateLuminance(pRGBRest)

        return round((L1 - L2)/255, 2)

# Function to measure social proof
def MeasureImageSocialProof(pImagePersonExists):
    if(pImagePersonExists == True):
        return -1
    else:
        return 1

# Function to measure present bias
def MeasurePresentBiasGap(pBenefitsText, pRisksText):

    # Tokenized text
    vTokenizedBenefitsText = word_tokenize(pBenefitsText)
    vTokenizedRisksText = word_tokenize(pRisksText)

    # Tagged text
    vTaggedBenefitsText = pos_tag(vTokenizedBenefitsText)
    vTaggedRisksText = pos_tag(vTokenizedRisksText)

    # Tenses for benefits
    vTenseBenefitsText = {}
    vTenseBenefitsText["Future"] = len([vWord for vWord in vTaggedBenefitsText if vWord[1] == "MD"])
    vTenseBenefitsText["Present"] = len([vWord for vWord in vTaggedBenefitsText if vWord[1] in ["VBP", "VBZ","VBG"]])
    vTenseBenefitsText["Past"] = len([vWord for vWord in vTaggedBenefitsText if vWord[1] in ["VBD", "VBN"]])
    vTenseBenefitsText["Total"] = len(pBenefitsText.split())
    vTaggedBenefitsTextScore = (int(vTenseBenefitsText["Past"]) * -1 +
                                int(vTenseBenefitsText["Present"]) * 0 +
                                int(vTenseBenefitsText["Future"]) * 1)/int(vTenseBenefitsText["Total"])

    # Tenses for risks
    vTenseRisksText = {}
    vTenseRisksText["Future"] = len([vWord for vWord in vTaggedRisksText if vWord[1] == "MD"])
    vTenseRisksText["Present"] = len([vWord for vWord in vTaggedRisksText if vWord[1] in ["VBP", "VBZ","VBG"]])
    vTenseRisksText["Past"] = len([vWord for vWord in vTaggedRisksText if vWord[1] in ["VBD", "VBN"]])
    vTenseRisksText["Total"] = len(pRisksText.split())
    vDominantTenseRisksTextScore = (int(vTenseRisksText["Past"]) * 1 +
                                    int(vTenseRisksText["Present"]) * 0 +
                                    int(vTenseRisksText["Future"]) * -1)/int(vTenseRisksText["Total"])

    vPresentBiasScore = round((vTaggedBenefitsTextScore + vDominantTenseRisksTextScore)/2, 2)

    return vPresentBiasScore

# Function to calculate luminescence
def CalculateLuminance(pRGBTuple):
    R, G, B = pRGBTuple
    return 0.2126 * R + 0.7152 * G + 0.0722 * B

# ------------------------------------------------------------ #
# ------------------------ HEART SCORE ----------------------- #
# ------------------------------------------------------------ #

# Calculating the HEART score
CalculateHEARTScore(vBenefitsText,
                    vRisksText,
                    vRGBBenefitsText,
                    vRGBRisksText,
                    vRGBBenefitsBackground,
                    vRGBRisksBackground,
                    vBenefitsVisibilityRatio,
                    vRisksVisibilityRatio,
                    vRGBImage,
                    vRGBRest,
                    vImagePersonExists)