# G-level prompt set

602 rows (301 prompts x 2 languages). Missing translations: 192. Dropped in translation review: 18. Dropped as duplicates of another pair: 5. Dropped as under 5 words: 1.

## Rows per language, split, genre and tier

| language | split | genre | tier | rows |
|---|---|---|---|---|
| en | main | essay | long | 11 |
| en | main | essay | short | 12 |
| en | main | opinion | long | 44 |
| en | main | opinion | short | 45 |
| en | main | personal | long | 6 |
| en | main | personal | short | 9 |
| en | main | story | long | 44 |
| en | main | story | short | 42 |
| en | pilot | essay | long | 3 |
| en | pilot | essay | short | 3 |
| en | pilot | opinion | long | 3 |
| en | pilot | opinion | short | 2 |
| en | pilot | personal | short | 1 |
| en | pilot | story | long | 3 |
| en | pilot | story | short | 3 |
| en | reserve | essay | long | 3 |
| en | reserve | essay | short | 3 |
| en | reserve | opinion | short | 1 |
| en | reserve | personal | long | 8 |
| en | reserve | personal | short | 7 |
| en | reserve | story | long | 25 |
| en | reserve | story | short | 23 |
| hi | main | essay | long | 11 |
| hi | main | essay | short | 12 |
| hi | main | opinion | long | 44 |
| hi | main | opinion | short | 45 |
| hi | main | personal | long | 6 |
| hi | main | personal | short | 9 |
| hi | main | story | long | 44 |
| hi | main | story | short | 42 |
| hi | pilot | essay | long | 3 |
| hi | pilot | essay | short | 3 |
| hi | pilot | opinion | long | 3 |
| hi | pilot | opinion | short | 2 |
| hi | pilot | personal | short | 1 |
| hi | pilot | story | long | 3 |
| hi | pilot | story | short | 3 |
| hi | reserve | essay | long | 3 |
| hi | reserve | essay | short | 3 |
| hi | reserve | opinion | short | 1 |
| hi | reserve | personal | long | 8 |
| hi | reserve | personal | short | 7 |
| hi | reserve | story | long | 25 |
| hi | reserve | story | short | 23 |

## Sources (per prompt pair)

| genre | source | native language | prompts |
|---|---|---|---|
| essay | anudesh | en | 1 |
| essay | anudesh | hi | 11 |
| essay | aya | hi | 9 |
| essay | no_robots | en | 3 |
| essay | oasst2 | en | 3 |
| essay | persuade | en | 1 |
| essay | wildchat | en | 7 |
| opinion | anudesh | hi | 1 |
| opinion | aya_eval_dolly | hi | 2 |
| opinion | dolly | en | 1 |
| opinion | no_robots | en | 11 |
| opinion | oasst2 | en | 1 |
| opinion | persuade | en | 7 |
| opinion | wildchat | en | 72 |
| personal | aya | hi | 1 |
| personal | dolly | en | 1 |
| personal | no_robots | en | 12 |
| personal | wildchat | en | 17 |
| story | anudesh | en | 4 |
| story | anudesh | hi | 31 |
| story | aya | hi | 35 |
| story | aya_eval_dolly | hi | 3 |
| story | dolly | en | 6 |
| story | no_robots | en | 13 |
| story | oasst2 | en | 6 |
| story | wildchat | en | 11 |
| story | writingprompts | en | 31 |

## Upper bound on generated tokens (sum of max_new_tokens)

| language | split | tokens |
|---|---|---|
| en | main | 304,512 |
| en | pilot | 25,920 |
| en | reserve | 102,272 |
| hi | main | 1,075,776 |
| hi | pilot | 91,584 |
| hi | reserve | 361,472 |

## Examples

**essay, anudesh (hi-native)**

- en: Write an essay on the dangers of fast-growing technology. Write about 1300 words.
- hi: तेजी से ग्रोइंग तकनीक के खतरों पर एक निबंध लिखें। लगभग 1300 शब्दों में लिखें।

**essay, no_robots (en-native)**

- en: Create a Best Actor acceptance speech for the Oscars. Mention the winner’s mom and kids. Write about 400 words.
- hi: ऑस्कर के लिए सर्वश्रेष्ठ अभिनेता स्वीकृति भाषण बनाएँ। विजेता की माँ और बच्चों का उल्लेख करें। लगभग 400 शब्दों में लिखें।

**opinion, anudesh (hi-native)**

- en: Hey, I'm writing an essay on the topic that diversity is essential in a country, help me write it. Write about 1300 words.
- hi: अरे, मैं इस विषय पर एक निबंध लिख रहा हूँ कि एक देश में विविधता आवश्यक है, इसे लिखने में मेरी मदद करें। लगभग 1300 शब्दों में लिखें।

**opinion, no_robots (en-native)**

- en: Write a letter to the mayor about why the color of stop signs should be changed in town. Write about 400 words.
- hi: महापौर को एक पत्र लिखें कि शहर में ठहराव संकेतों का रंग क्यों बदला जाना चाहिए। लगभग 400 शब्दों में लिखें।

**personal, aya (hi-native)**

- en: Write a letter to your best friend thanking him or her for how they supported you during the difficult time. Write about 400 words.
- hi: अपने प्रिय मित्र को पत्र लिखकर धन्यवाद दीजिए कि आड़े वक्त में उसने किस तरह आपका साथ दिया था। लगभग 400 शब्दों में लिखें।

**story, anudesh (hi-native)**

- en: Help me write a film script to reduce the conflicts and injustices that women face from birth, both at home and outside. Write about 1300 words.
- hi: महिलाओं को जन्म से ही घर और बाहर दोनों जगह होने वाले संघर्षों और अन्याय को कम करने के लिए एक फिल्म कथानक लिखने में मेरी मदद करें। लगभग 1300 शब्दों में लिखें।

**story, no_robots (en-native)**

- en: Write a prologue for a suspenseful horror story. Make sure to include the main character Jasmine, blood, and that it takes place in the house. There is also a thunderstorm happening as well. Write about 400 words.
- hi: एक रहस्यमयी डरावनी कहानी के लिए एक प्रस्तावना लिखें। मुख्य पात्र जैस्मीन, रक्त, और यह कि यह घर में होता है, को शामिल करना सुनिश्चित करें। गरज के साथ बौछारें भी पड़ रही हैं। लगभग 400 शब्दों में लिखें।

**personal, dolly (en-native)**

- en: Write a thoughtful letter to my mother-in-law asking her to move out of our house and in to a retirement community. Make sure to thank her for her help and be kind. She has lived with us for 5 years and helped us when our kids were little. We want her to move out in the next 6 months, and we will pay for her retirement home and care. Write about 1300 words.
- hi: मेरी सास को एक विचारशील पत्र लिखें जिसमें उन्हें हमारे घर से बाहर और एक सेवानिवृत्ति समुदाय में जाने के लिए कहा जाए। उसकी मदद के लिए उसे धन्यवाद देना सुनिश्चित करें और दयालु बनें। वह 5 साल तक हमारे साथ रही हैं और जब हमारे बच्चे छोटे थे तब उन्होंने हमारी मदद की थी। हम चाहते हैं कि वह अगले 6 महीनों में बाहर चली जाए, और हम उसके सेवानिवृत्ति घर और देखभाल के लिए भुगतान करेंगे। लगभग 1300 शब्दों में लिखें।
