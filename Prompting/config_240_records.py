class Config:

    file_path = "D:\OVGU _Saurabh\SEM 3\Review Project\Code\Datasets\sampled_deceptive_opinion_new.csv"

    model = "llama3.2:1b"

    # Define zero-shot classification prompt
    # text placeholder in the prompt template will be replaced with actual review text.
    zero_shot_prompt_template = """
    You are an expert at detecting fake and deceptive reviews.

    Task: Classify the following review of hotel or a product that is listed on amazon as either "truthful" or "deceptive". Analyze the language and structure of the review,
	It should not be overly generic or templeted language,should not have sales-like wording.

    Review: "{text}"

    Classify this review as either "truthful" or "deceptive" (respond with only one word):"""

    # Define one-shot classification prompt with one example
    one_shot_prompt_template = """
     You are an expert at detecting truthful and deceptive reviews.

    Task: Classify the following review of hotel or a product that is listed on amazon as either "truthful" or "deceptive". Analyze the language and structure of the review,
	It should not be overly generic or templeted language,should not have sales-like wording and lacks details or specificity.

    Example:
    Review: "As a former Chicagoan, I'm appalled at the Amalfi Hotel Chicago. First of all, I was expecting luxury and hospitality, neither of which I received. There's an Experience Designer who is supposed to be like a 'personal concierge,' but my experience with my ED was terrible. I felt like he was trying to pressure me into staying more days than I wanted to. Not only that, but I couldn't understand what he was saying most of the time because he was talking so fast. When I finally got to my room, I was disappointed with the quality of the furniture and the room's cleanliness. I had to ask for a maid to come and give me clean towels because some of the towels in the bathroom were damp. On top of that, the bed was messily done; I could have done a better job on my own bed at home. I was angry at this point, because I was paying a lot of money for every night I was staying at Amalfi, and I didn't expect to be greeted with wet towels. I needed to use the Wi-Fi to download some important documents, and the internet was surprisingly slow. Even a very basic hotel or motel could have offered better, maybe even faster internet access. When I finally checked out of the Amalfi, I made sure that my supposed personal concierge knew all of the problems I'd had with my room and the hotel. I was glad to see the Amalfi getting smaller in the mirror as I drove away!
    "
    Classification: deceptive

    Now classify this review:
    Review: "{text}"
    Classify this review as either "truthful" or "deceptive" (respond with only one word):"""

    # Define few-shot classification prompt with multiple examples
    few_shot_prompt_template = """
    You are an expert at detecting truthful and deceptive reviews like ReviewMeta.


    Task: Classify the following review of hotel or a product that is listed on amazon as either "truthful" or "deceptive". Analyze the language and structure of the review,
	It should not be overly generic or templeted language,should not have sales-like wording and lacks details or specificity.

    Here are some examples:

    Example 1:
    Review: "As a former Chicagoan, I'm appalled at the Amalfi Hotel Chicago. First of all, I was expecting luxury and hospitality, neither of which I received. There's an Experience Designer who is supposed to be like a 'personal concierge,' but my experience with my ED was terrible. I felt like he was trying to pressure me into staying more days than I wanted to. Not only that, but I couldn't understand what he was saying most of the time because he was talking so fast. When I finally got to my room, I was disappointed with the quality of the furniture and the room's cleanliness. I had to ask for a maid to come and give me clean towels because some of the towels in the bathroom were damp. On top of that, the bed was messily done; I could have done a better job on my own bed at home. I was angry at this point, because I was paying a lot of money for every night I was staying at Amalfi, and I didn't expect to be greeted with wet towels. I needed to use the Wi-Fi to download some important documents, and the internet was surprisingly slow. Even a very basic hotel or motel could have offered better, maybe even faster internet access. When I finally checked out of the Amalfi, I made sure that my supposed personal concierge knew all of the problems I'd had with my room and the hotel. I was glad to see the Amalfi getting smaller in the mirror as I drove away!"
    Classification: deceptive

    Example 2:
    Review: "We stayed for a one night getaway with family on a thursday. Triple AAA rate of 173 was a steal. 7th floor room complete with 44in plasma TV bose stereo, voss and evian water, and gorgeous bathroom(no tub but was fine for us) Concierge was very helpful. You cannot beat this location... Only flaw was breakfast was pricey and service was very very slow(2hours for four kids and four adults on a friday morning) even though there were only two other tables in the restaurant. Food was very good so it was worth the wait. I would return in a heartbeat. A gem in chicago..."
    Classification: truthful

    Example 3:
    Review: "The Palmer House Hilton, while it looks good in pictures, and the outside, is actually a disaster of a hotel. When I went through, the lobby was dirty, my room hadn't been cleaned, and smelled thoroughly of smoke. When I requested more pillows, the lady on the phone scoffed at me and said she'd send them up. It took over an hour for 2 pillows. This hotel is a good example that what you pay for isn't always what you get. I will not be returning."
    Classification: deceptive

    Example 4:
    Review: "Triple A rate with upgrade to view room was less than $200 which also included breakfast vouchers. Had a great view of river, lake, Wrigley Bldg. & Tribune Bldg. Most major restaurants, Shopping, Sightseeing attractions within walking distance. Large room with a very comfortable bed."
    Classification: truthful

    Now classify this review:
    Review: "{text}"
    Classify this review as either "truthful" or "deceptive" (respond with only one word):"""

