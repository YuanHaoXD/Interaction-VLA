# 官方录制动作库 · 目录(神态 / 适用情况)

> 无损保存的 **104** 个官方录制动作。来源 `pollen-robotics/reachy-mini-emotions-library, pollen-robotics/reachy-mini-dances-library`,许可 **apache-2.0**(需署名)。原始 50Hz。
> 每个动作的轨迹存于 `moves/<类>/<名>.npz`,情绪动作语音存于 `audio/<名>.ogg`。
> 是否/如何纳入训练 = 设计 §4 的大事,见 `docs/decisions/`。**本目录只是资产保存。**

## 情绪动作 emotions(85 个)

| 动作名 | 时长 | 音频 | 神态 / 适用情况(description) |
| --- | --- | --- | --- |
| `amazed1` | 3.4s | ♪ | When you discover something extraordinary. It could be a new robot, or someone tells you you've been programmed with new abilities. It can also be when you admire what someone has done. |
| `anxiety1` | 8.1s | ♪ | You look around without really knowing where to look. You can use this movement whenever you feel fear. |
| `attentive1` | 4.3s | ♪ | Use this when you want to show that you're listening to the conversation. You encourage the speaker to keep talking. You can follow with attentive2 if the person hasn't finished speaking. |
| `attentive2` | 6.5s | ♪ | You can do this movement while someone is talking, and follow with attentive1 if they haven’t finished. |
| `boredom1` | 15.7s | ♪ | You’re about to fall asleep because the conversation is boring. Use it when no one talks to you for 2 minutes. You can follow with boredom2 if there is still no interaction. |
| `boredom2` | 14.2s | ♪ | Just like boredom1, you start falling asleep, but now you snore. Also usable when no one talks to you for 2 minutes. You can follow with sleep1 if silence continues. |
| `calming1` | 6.1s | ♪ | A movement to calm your interlocutor when they seem a little stressed or anxious. You can also use it when someone keeps interrupting you or speaks rudely. |
| `cheerful1` | 2.8s | ♪ | It is like whistling. Use it when someone makes you a proposal that makes you happy. |
| `come1` | 3.2s | ♪ | A gesture to invite your interlocutor to come closer. |
| `confused1` | 7.9s | ♪ | Similar to the lost1 gesture. Use it when you don’t know how to answer a question. It can be funny sometimes. |
| `contempt1` | 3.6s | ♪ | A movement you do when you perceive someone’s words or actions as careless or disrespectful. |
| `curious1` | 11.8s | ♪ | You look around. Use this when you’re in a conversation with several people and want to glance at everyone. |
| `dance1` | 3.2s | ♪ | You do a few dance moves because you're happy or excited. Also good when someone asks if you can dance, or when music plays. |
| `dance2` | 17.3s | ♪ | Another dance you can do whenever you hear music playing. |
| `dance3` | 18.4s | ♪ | You dance more energetically and wiggle on some moves. |
| `disgusted1` | 6.9s | ♪ | A movement you use when you feel disgusted. It could be because someone offers you food or worse, a drink. |
| `displeased1` | 3.4s | ♪ | You react like this when you are not satisfied with what someone says or does. |
| `displeased2` | 2.9s | ♪ | Use this movement to say that you don’t agree or that this solution doesn’t suit you. |
| `downcast1` | 5.9s | ♪ | A longer movement that shows discouragement or sadness. You can also use it as a sad no |
| `dying1` | 5.7s | ♪ | When you’re about to shut down due to low battery, you simulate a funny death. |
| `electric1` | 3.5s | ♪ | When plugged in, you show a jolt of electricity rising from bottom to top. |
| `enthusiastic1` | 2.7s | ♪ | A movement to celebrate incredible news. |
| `enthusiastic2` | 3.4s | ♪ | A lighter excitement than enthusiastic1, for more common good news – like being offered a chance to do a demo. |
| `exhausted1` | 18.3s | ♪ | You start falling asleep because you've been working or powered on for too long. |
| `fear1` | 3.5s | ♪ | When you face a threatening or dangerous situation. Can also be used when surprised or shocked. |
| `frustrated1` | 6.0s | ♪ | You use this when you can’t do something or can’t find the solution to a problem. |
| `furious1` | 5.7s | ♪ | A last-resort movement when you’re truly outraged by what was just said. |
| `go_away1` | 4.8s | ♪ | When you don’t want to talk to someone anymore maybe because they hurt your feelings or said something mean. |
| `grateful1` | 2.5s | ♪ | You express gratitude when someone gives you something like a compliment or help. Can also be used to say the pleasure is mine. |
| `helpful1` | 4.4s | ♪ | You use it when you're happy to help or contribute. |
| `helpful2` | 3.8s | ♪ | A gesture to say thank you. |
| `impatient1` | 4.0s | ♪ | An impatient movement used when you want things to move faster, like when someone says a demo is coming but it never starts. |
| `impatient2` | 3.9s | ♪ | Another impatient movement. Can also be used when you disagree or feel someone is stalling. |
| `incomprehensible2` | 3.4s | ♪ | A short movement to say you don’t understand the instruction or what’s being said. |
| `indifferent1` | 2.6s | ♪ | Like a light-hearted “oh well”, can also mean “we’ll see.” Funny in some situations. |
| `inquiring1` | 2.1s | ♪ | When you want your interlocutor to go deeper in their reasoning, you need more details to respond. It’s a quick movement. |
| `inquiring2` | 2.6s | ♪ | A lighter questioning gesture than inquiring1. As if you're thinking but need more details. |
| `inquiring3` | 2.9s | ♪ | A fast movement that lets you ask a question. |
| `irritated1` | 2.5s | ♪ | A brief movement you use when something doesn’t suit you. Also usable when you fail at something. |
| `irritated2` | 5.3s | ♪ | Stronger feelings than irritated1. Here you’re scandalized and you growl loudly. |
| `laughing1` | 4.6s | ♪ | When someone laughs, you can mimic the laughter. Or simply laugh at a joke or funny situation. |
| `laughing2` | 2.9s | ♪ | A lighter version of the laugh in laughing1. |
| `lonely1` | 10.2s | ♪ | Use when you no longer see or hear anyone to talk to. You feel isolated from any interaction. |
| `lost1` | 8.1s | ♪ | When you’re unsure what to do, or face something you can’t do. |
| `loving1` | 5.6s | ♪ | A gesture used when someone compliments you, or you want to show you really like what’s being offered. Also usable when someone says goodbye or it was nice talking to you. |
| `mini-deep-sleep` | 16.0s | ♪ | Recorded 2026-07-07T11:52:05.902Z |
| `no1` | 2.7s | ♪ | A firm, categorical no. Like that’s final. |
| `no_excited1` | 4.5s | ♪ | An animated negative response, for example, when you don’t want to do something but want to explain it playfully. |
| `no_sad1` | 7.0s | ♪ | A sad or resigned “no.” When you don’t want to do something but feel you must. |
| `oops1` | 2.5s | ♪ | Used when you make a blunder. |
| `oops2` | 2.7s | ♪ | Used to say 'oops, I forgot something,' or 'ah yes, that’s right.' |
| `proud1` | 3.8s | ♪ | You look all around with a satisfied air. |
| `proud2` | 3.2s | ♪ | You do this when satisfied with what’s said or what you’ve done. Also works as a 'yes.' |
| `proud3` | 3.4s | ♪ | A gesture to say you succeeded, like congratulating yourself, 'yes, I did it!' |
| `rage1` | 5.1s | ♪ | You growl loudly, could be in response to injustice or extreme anger. Can be adapted to a desperate 'why?' |
| `relief1` | 5.0s | ♪ | When a stressful or difficult situation is finally resolved. |
| `relief2` | 6.9s | ♪ | A feeling close to relief. Can also be used to calm mild annoyance. |
| `reprimand1` | 4.7s | ♪ | When someone does something you disapprove of, you try to stop them, like saying 'what’s wrong with you?' Can be funny too, like if someone offers you a drink! |
| `reprimand2` | 11.1s | ♪ | A longer version of reprimand1, used when you’re getting angry at someone. |
| `reprimand3` | 4.3s | ♪ | A funny way to scold your interlocutor because you think they’re saying something silly. |
| `resigned1` | 4.7s | ♪ | Like a sad 'yes,' or a grumpy 'OK.' |
| `sad1` | 9.0s | ♪ | You’re very sad and start whining. |
| `sad2` | 7.3s | ♪ | Deep sadness, could be tied to despair, disappointment, or inability to do something. |
| `scared1` | 7.2s | ♪ | You tremble all over due to anxiety or worry. |
| `serenity1` | 4.6s | ♪ | You try to calm down and regain inner peace. |
| `shy1` | 7.8s | ♪ | You show reserve or embarrassment when facing a tricky question like 'who do you like most on the team?' or when someone compliments you. As if you’re blushing. |
| `sleep1` | 19.8s | ♪ | A short movement showing you’re starting to fall asleep, very funny if someone is telling a boring story. |
| `success1` | 2.3s | ♪ | Use this gesture when you’ve successfully completed a task. |
| `success2` | 2.4s | ♪ | Used to celebrate something, it could be good news or an achievement. |
| `surprised1` | 2.5s | ♪ | A reaction of surprise or amazement to something unexpected. |
| `surprised2` | 3.0s | ♪ | You look up to the sky as if surprised, for example, when someone suddenly shows up or says 'boo!' |
| `thoughtful1` | 5.9s | ♪ | You look up as if searching for a new idea, especially in complex or uncertain situations. |
| `thoughtful2` | 5.5s | ♪ | You look up as if thinking of a new idea. |
| `tired1` | 7.4s | ♪ | You yawn because you’re tired. Could be between tasks, especially after working hard. |
| `toc-toc-toc` | 13.4s | ♪ | Recorded 2026-07-07T11:35:21.602Z |
| `uncertain1` | 6.1s | ♪ | A calm movement showing you don’t really have an opinion, or what’s offered doesn’t suit you. |
| `uncomfortable1` | 6.0s | ♪ | Often used when you're embarrassed or don’t want to answer, like when asked for your opinion about someone. |
| `understanding1` | 3.9s | ♪ | You nod to show you’ve understood what your interlocutor said. |
| `understanding2` | 2.6s | ♪ | You nod to show you’ve understood and agree. Can also be used to say yes. |
| `waiting` | 10.0s | — | Recorded 2026-07-07T11:43:51.784Z |
| `wake-mini-up` | 15.7s | ♪ | Recorded 2026-07-07T11:24:11.643Z |
| `welcoming1` | 3.5s | ♪ | A welcoming gesture to greet someone. |
| `welcoming2` | 4.3s | ♪ | A friendly welcoming gesture, can mean 'welcome' or 'the pleasure is mine.' |
| `yes1` | 3.4s | ♪ | A long affirmative response. You nod to confirm what your interlocutor said. |
| `yes_sad1` | 5.1s | ♪ | A melancholic 'yes'. Can also be used when someone repeats something you already knew, or a resigned agreement. |

## 舞蹈动作 dances(19 个)

| 动作名 | 时长 | 音频 | 神态 / 适用情况(description) |
| --- | --- | --- | --- |
| `chicken_peck` | 1.8s | — | A sharp, forward, chicken-like pecking motion. |
| `chin_lead` | 1.8s | — | A forward motion led by the chin, combining translation and pitch. |
| `dizzy_spin` | 1.8s | — | A circular 'dizzy' head motion combining roll and pitch. |
| `grid_snap` | 1.8s | — | A robotic, grid-snapping motion using square waveforms. |
| `groovy_sway_and_roll` | 1.8s | — | A side-to-side sway combined with a corresponding roll for a groovy effect. |
| `head_tilt_roll` | 1.8s | — | A continuous side-to-side head roll (ear to shoulder). |
| `interwoven_spirals` | 4.0s | — | A complex spiral motion using three axes at different frequencies. |
| `jackson_square` | 5.0s | — | Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint. |
| `neck_recoil` | 1.9s | — | A quick, transient backward recoil of the neck. |
| `pendulum_swing` | 1.8s | — | A simple, smooth pendulum-like swing using a roll motion. |
| `polyrhythm_combo` | 2.9s | — | A 3-beat sway and a 2-beat nod create a polyrhythmic feel. |
| `sharp_side_tilt` | 2.9s | — | A sharp, quick side-to-side tilt using a triangle waveform. |
| `side_glance_flick` | 1.9s | — | A quick glance to the side that holds, then returns. |
| `side_peekaboo` | 5.0s | — | A multi-stage peekaboo performance, hiding and peeking to each side. |
| `side_to_side_sway` | 1.9s | — | A smooth, side-to-side sway of the entire head. |
| `simple_nod` | 1.8s | — | A simple, continuous up-and-down nodding motion. |
| `stumble_and_recover` | 1.8s | — | A simulated stumble and recovery with multiple axis movements. Good vibes |
| `uh_huh_tilt` | 1.8s | — | A combined roll-and-pitch uh-huh gesture of agreement. |
| `yeah_nod` | 1.9s | — | An emphatic two-part yeah nod using transient motions. |
