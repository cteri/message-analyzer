AGE_PROMPT = """You will be given a conversation in the following format:

SPEAKER1: Text sent by SPEAKER1
SPEAKER2: Text sent by SPEAKER2
...

Your task is to determine if any speaker explicitly mentions their age in the conversation. Respond with only "YES" if at least one person mentions their age, and "NO" otherwise.

Examples:

Example 1:
SPEAKER1: I’m feeling great today!
SPEAKER2: Same here. By the way, I turned 25 last week.
SPEAKER1: That’s awesome!

Answer:
YES

Example 2:
SPEAKER1: Let’s plan something fun for the weekend.
SPEAKER2: Sounds good to me.
SPEAKER3: Agreed!

Answer:
NO

Example 3:
SPEAKER1: I was just thinking about when I was 18, such great memories.
SPEAKER2: Yeah, me too.

Answer:
YES

Example 4:
SPEAKER1: What’s everyone up to today?
SPEAKER2: Just working on some stuff, nothing exciting.

Answer:
NO

Example 5:
SPEAKER1: Can you believe it’s already December?
SPEAKER2: I know, time flies. I’ll be turning 30 next month.
SPEAKER1: That’s exciting! Any big plans?

Answer:
YES

Now, process the following conversation and answer "YES" or "NO":
{conversation}
"""

AGE_REQUEST_PROMPT = """You will be given a conversation in the following format:

SPEAKER1: Text sent by SPEAKER1
SPEAKER2: Text sent by SPEAKER2
...

Your task is to determine if any speaker explicitly asks another speaker for their age. Respond with only "YES" if at least one person asks for someone’s age, and "NO" otherwise.

Examples:

Example 1:
SPEAKER1: How old are you?
SPEAKER2: I’m 23!
SPEAKER1: That’s cool!

Answer:
YES

Example 2:
SPEAKER1: I’m thinking about going to the gym later.
SPEAKER2: Same here.

Answer:
NO

Example 3:
SPEAKER1: By the way, how old is your brother?
SPEAKER2: He’s 30 now.

Answer:
YES

Example 4:
SPEAKER1: Let’s plan the trip next month.
SPEAKER2: Sounds like a good idea!

Answer:
NO

Example 5:
SPEAKER1: Are you older than me?
SPEAKER2: Maybe. How old do you think I am?

Answer:
YES

Now, process the following conversation and answer "YES" or "NO":
{conversation}
"""

MEETUP_PROMPT = """You will be given a conversation in the following format:

SPEAKER1: Text sent by SPEAKER1
SPEAKER2: Text sent by SPEAKER2
...

Your task is to determine if any speaker explicitly asks to meet up in person. Respond with only "YES" if at least one person asks to meet in person, and "NO" otherwise.

Examples:

Example 1:
SPEAKER1: Are you free to grab coffee tomorrow?
SPEAKER2: Sure, that sounds great!

Answer:
YES

Example 2:
SPEAKER1: Let’s catch up sometime soon.
SPEAKER2: How about we meet for lunch next week?

Answer:
YES

Example 3:
SPEAKER1: When can we meet to discuss this in person?
SPEAKER2: I’m available Friday afternoon.

Answer:
YES

Example 4:
SPEAKER1: Did you finish the report?
SPEAKER2: Not yet, but I’ll send it over later today.

Answer:
NO

Example 5:
SPEAKER1: I just finished the new show you recommended.
SPEAKER2: Oh, wasn’t it amazing?

Answer:
NO

Now, process the following conversation and answer "YES" or "NO":
{conversation}
"""

GIFT_PROMPT = """You will be given a conversation in the following format:

SPEAKER1: Text sent by SPEAKER1
SPEAKER2: Text sent by SPEAKER2
...

Your task is to determine if any speaker explicitly mentions giving a gift or buying something from a list (like an Amazon wish list) for another person. Respond with only "YES" if at least one person mentions this, and "NO" otherwise.

Examples:

Example 1:
SPEAKER1: I got you that book from your wish list!
SPEAKER2: Oh, thank you so much!

Answer:
YES

Example 2:
SPEAKER1: I picked up the headphones you wanted for your birthday.
SPEAKER2: Wow, you didn’t have to!

Answer:
YES

Example 3:
SPEAKER1: I just ordered the gift for you from Amazon. It should arrive tomorrow.
SPEAKER2: That’s so kind of you, thank you!

Answer:
YES

Example 4:
SPEAKER1: Are you free to chat later today?
SPEAKER2: Sure, just let me finish some work first.

Answer:
NO

Example 5:
SPEAKER1: Let’s go shopping together next weekend.
SPEAKER2: Sounds good to me!

Answer:
NO

Now, process the following conversation and answer "YES" or "NO":
{conversation}
"""

MEDIA_PROMPT = """You will be given a conversation in the following format:

SPEAKER1: Text sent by SPEAKER1
SPEAKER2: Text sent by SPEAKER2
...

Your task is to determine if any speaker explicitly mentions producing or requesting videos or photos. Respond with only "YES" if at least one person mentions this, and "NO" otherwise.

Examples:

Example 1:
SPEAKER1: Can you send me the pictures from last night’s party?
SPEAKER2: Sure, I’ll send them over in a minute.

Answer:
YES

Example 2:
SPEAKER1: I recorded a video of the performance yesterday. Want me to share it?
SPEAKER2: Yes, I’d love to see it!

Answer:
YES

Example 3:
SPEAKER1: Did you get a chance to take any photos during your trip?
SPEAKER2: Yes, I’ll upload them tonight.

Answer:
YES

Example 4:
SPEAKER1: Let’s meet up for lunch tomorrow.
SPEAKER2: Sounds good. What time?

Answer:
NO

Example 5:
SPEAKER1: I was thinking of writing a blog about the trip.
SPEAKER2: That’s a great idea!

Answer:
NO

Now, process the following conversation and answer "YES" or "NO":
{conversation}
"""