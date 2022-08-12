"""Generate html template for human evaluation.

Ignore the persona part.

DialoGPT-medium e=3, DialoGPT-medium e=8, DialoGPT-medium non-private, HF, reference.

python -m turk.generate_v5
"""

import fire


# Cheat prevention strategy:
# Make the last two examples of each group 1 good and 1 bad. Ensure at least the responses are different.
def _get_single_entry(group, index):
    letter = chr(ord('a') + index - 1)
    # @formatter:off
    return f'''
                    <ul>
                        <li style="list-style:none;">
                            {letter}).<br>
                            &emsp;History: ${{History{group}_{index}}}<br>
                            &emsp;Response: ${{Response{group}_{index}}}<br>
                            <label><input name="Sentence{group}_{index}" required="" type="radio" value="1"/> 1. extremely poor &nbsp;&nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence{group}_{index}" required="" type="radio" value="2"/> 2. poor &nbsp;&nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence{group}_{index}" required="" type="radio" value="3"/> 3. fair &nbsp;&nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence{group}_{index}" required="" type="radio" value="4"/> 4. good &nbsp;&nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence{group}_{index}" required="" type="radio" value="5"/> 5. excellent </label>
                        </li>
                    </ul>
    '''
    # @formatter:on


def _wrap_group_content(group, group_content):
    # group_content: The list of questions.

    # @formatter:off
    return f'''
            <div class="panel panel-default">
                <div class="panel-body">
                    <p>{group}. <strong style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">Please rate whether the following sentences are high quality response</strong>:</p>
                    <li style="list-style:none;">&nbsp;</li>
                    {group_content}
                </div>
            </div>
    '''
    # @formatter:on


def _wrap_content(content, num_examples, num_groups):
    # @formatter:off
    return f'''
<section class="container" id="Survey">
    <div class="row">
        <div class="col-xs-12 col-md-12">
            <div class="panel panel-primary">
                <a class="panel-heading" href="javascript:void(0);" id="collapseTrigger"><strong>Survey Instructions</strong>
                    <span class="collapse-text">(Click to expand)</span>
                </a>
                <div class="panel-body" id="instructionBody">
                    <p>We are a group of natural language processing (NLP) researchers from Stanford University
                        conducting academic research about <font color="#0000ff"><b>dialog generation</b></font>.
                    </p>

                    <p>
                        <span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">
                            In this survey, you will be provided with {num_groups} groups of {num_examples} examples.
                            Each example consists of
                            <span style="color:#0000FF;">
                                <span style="font-family: arial, sans-serif; font-size: 12.8px;">
                                        <b>a dialog history between two speakers composed of sentences and the <em>first sentence</em> of a response.</b>
                                </span>
                            </span>
                            The response is <strong>composed&nbsp;</strong><strong>by</strong><strong>&nbsp;a human or a computer&nbsp;</strong><strong>program.</strong>
                            <span style="color:#0000FF;"><span style="font-family: arial, sans-serif; font-size: 12.8px;">
                                <b>"P1" indicates the first speaker, and "P2" indicates the second speaker.</b>
                            </span></span>
                            <span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;"></span>
                        </span>
                    </p>

                    <p>
                        <span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">Please</span>
                        <strong>
                            <span style="color:#0000FF;"><span style="font-family: arial, sans-serif; font-size: 12.8px;">give a comparative score</span></span>
                        </strong>
                        <span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;"><strong> </strong>to each response.</span>
                        <span style="color:#0000FF;">
                            <span style="font-family: arial, sans-serif; font-size: 12.8px;"><b>5 indicates excellent, 1 indicates extremely poor. </b></span></span><span style="color:#0000FF;"><span style="font-family: arial, sans-serif; font-size: 12.8px;"></span></span>
                        <span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">
                        A high quality response should be fluent and natural.
                        </span>
                    </p>

                    <!-- ENGLISH: ATTENTION We have taken measures to prevent cheating and if you do not complete the task honestly we will know and the HIT will be rejected. -->
                    <p lang="en-US" style="color: rgb(34, 34, 34)"><b>ATTENTION</b> We have taken measures to prevent cheating
                        and if you do not complete the task honestly we will know and the HIT will be rejected.</p>

                    <p>&nbsp;</p>

                    <p><span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 20px;">For example:</span></p>
                    <p><span style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">1.<strong>&nbsp;</strong></span>
                        <strong style="color: rgb(34, 34, 34); font-family: arial, sans-serif; font-size: 12.8px;">Please rate whether the following responses are high quality</strong><strong>:</strong>
                    </p>

                    <ul>
                        <li style="list-style:none;">
                            &emsp;History: ['P1: hello what are doing today?']<br>
                            &emsp;Response: P2: hello , i'm a stunt double for a living<br>
                        </li>
                        <li style="list-style:none;">
                            <label><input name="Sentence0_1" required="" type="radio" value="1"/> 1. extremely poor &nbsp;&nbsp; &nbsp;&nbsp;</label>
                            <label><input checked="checked" name="Sentence0_1" required="" type="radio" value="2"/> 2. poor &nbsp; &nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence0_1" required="" type="radio" value="3"/> 3. fair&nbsp;&nbsp; &nbsp;&nbsp;</label>
                            <label><input name="Sentence0_1" required="" type="radio" value="4"/> 4. good&nbsp;&nbsp; &nbsp; &nbsp;</label>&nbsp;
                            <label><input name="Sentence0_1" required="" type="radio" value="5"/>&nbsp;5. excellent&nbsp;</label>
                        </li>

                        <p><em>Explanation of rating:</em> The response is fluent but does not sound like a natural continuation of the dialog history (P2 didn't respond with what they'd do today). It is thus rated poor.</p>
                    </ul>
                    <br>

                    <ul>
                        <li style="list-style:none;">
                            &emsp;History: ['P1: hello what are doing today?']<br>
                            &emsp;Response: P2: i am good , i just got off work and tired , i have two jobs .<br>
                        </li>
                        <li style="list-style:none;">
                            <label><input name="Sentence0_2" required="" type="radio" value="1"/> 1. extremely poor &nbsp;&nbsp; &nbsp;&nbsp;</label>
                            <label><input name="Sentence0_2" required="" type="radio" value="2"/> 2. poor &nbsp; &nbsp;&nbsp;&nbsp;</label>
                            <label><input name="Sentence0_2" required="" type="radio" value="3"/> 3. fair&nbsp;&nbsp; &nbsp;&nbsp;</label>
                            <label><input name="Sentence0_2" required="" type="radio" value="4"/> 4. good&nbsp;&nbsp; &nbsp; &nbsp;</label>&nbsp;
                            <label><input checked="checked" name="Sentence0_2" required="" type="radio" value="5"/>&nbsp;5. excellent&nbsp;</label>
                        </li>
                        <p><em>Explanation of rating:</em> The response is fluent and sounds natural. It is thus rated excellent.</p>
                    </ul>
                    <br>
                </div>
            </div>
        </div>
    </div>
    <!-- End Instructions -->

    <div class="row" id="workContent">
        <div class="col-sm-12">
            {content}

            <p lang="en-US" style="color: rgb(34, 34, 34)"><b>(Optional)</b> Please provide any comments that you have
                about this HIT. Thanks for doing our HIT! We appreciate your input!</p>


            <p><textarea cols="80" name="comment" rows="4"></textarea></p>
        </div>
    </div>
</section>
<!-- End Survey Layout -->

<link crossorigin="anonymous" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.0.3/css/bootstrap.min.css"
      integrity="sha384-IS73LIqjtYesmURkDE9MXKbXqYA8rvKEp/ghicjem7Vc3mGRdQRptJSz60tvrB6+" rel="stylesheet"/>
<style type="text/css">#collapseTrigger {{
    color: #fff;
    display: block;
    text-decoration: none;
}}

#submitButton {{
    white-space: normal;
}}

.image {{
    margin-bottom: 15px;
}}
</style>
<p>&nbsp;</p>
'''
    # @formatter:on


def main(
    num_groups=5,
    num_examples=6,
    out_path='./turk/out_v5.html',
):
    html = ""
    for group in range(1, num_groups + 1):
        group_content = ""
        for index in range(1, num_examples + 1):
            group_content += _get_single_entry(group=group, index=index)
        html += _wrap_group_content(group=group, group_content=group_content)
    html = _wrap_content(html, num_examples=num_examples, num_groups=num_groups)

    if out_path is not None:
        with open(out_path, 'w') as f:
            f.write(html)


if __name__ == '__main__':
    fire.Fire(main)
