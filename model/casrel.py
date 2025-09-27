class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path,
                                            checkpoint_path=checkpoint_path,
                                            model='bert',
                                            vocab_size=21128,
                                            hidden_size=768,
                                            num_hidden_layers=12,
                                            num_attention_heads=12,
                                            intermediate_size=3072,
                                            hidden_act="gelu")
        self.head = nn.Linear(768, 768)
        self.tail = nn.Linear(768, 768)
        # refined word representation
        self.co_layers = 1
        self.crossatt1 = CoAttention(768, num_attention_heads=8, output_attentions=8,
                                     attention_probs_dropout_prob=0.3)
        self.crossatt2 = CoAttention(768,  num_attention_heads=8, output_attentions=8,
                                     attention_probs_dropout_prob=0.3)

        self.linear1 = nn.Linear(768, 2)
        self.condLayerNorm = LayerNorm(hidden_size=768, conditional_size=768 * 2)
        self.linear2 = nn.Linear(768, len(predicate2id) * 2)

    @staticmethod
    def extract_subject(inputs):
        """Extract vector representation of subject from the output according to subject_ids
        """
        output, subject_ids = inputs
        start = torch.gather(output, dim=1, index=subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, dim=1, index=subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]

    def forward(self, *inputs):
        # predict subject
        seq_output = self.bert(inputs[:2])  # [batch_size, seq_len, hidden_size]
        mask_token_ids = inputs[0].gt(0).long()
        seq_output_head = self.head(seq_output)
        seq_output_tail = self.tail(seq_output)
        tmp_head, tmp_tail = seq_output_head, seq_output_tail

        for _ in range(0, self.co_layers):
            cq_biatt_output = self.crossatt1(seq_output_head, seq_output_tail, mask_token_ids)
            qc_biatt_output = self.crossatt2(seq_output_tail, seq_output_head, mask_token_ids)

            seq_output_tail = cq_biatt_output
            seq_output_head = qc_biatt_output
        seq_output_head += tmp_head
        seq_output_tail += tmp_tail

        subject_preds = (torch.sigmoid(self.linear1(seq_output_head))) ** 2  # [batch_size, seq_len, 2]

        # input subject, predict object
        # Incorporate subject into prediction of object through Conditional Layer Normalization
        subject_ids = inputs[2]
        subject = self.extract_subject([seq_output_tail, subject_ids])
        output = self.condLayerNorm(seq_output_tail, subject)
        output = (torch.sigmoid(self.linear2(output))) ** 4
        object_preds = output.reshape(*output.shape[:2], len(predicate2id), 2)

        return [subject_preds, object_preds]

    def predict_subject(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output = self.bert(inputs[:2])  # [batch_size, seq_len, hidden_size]
            mask_token_ids = inputs[0].gt(0).long()
            seq_output_head = self.head(seq_output)
            seq_output_tail = self.tail(seq_output)
            tmp_head, tmp_tail = seq_output_head, seq_output_tail

            for _ in range(0, self.co_layers):
                cq_biatt_output = self.crossatt1(seq_output_head, seq_output_tail, mask_token_ids)
                qc_biatt_output = self.crossatt2(seq_output_tail, seq_output_head, mask_token_ids)

                seq_output_tail = cq_biatt_output
                seq_output_head = qc_biatt_output
            seq_output_head += tmp_head
            seq_output_tail += tmp_tail

            subject_preds = (torch.sigmoid(self.linear1(seq_output_head))) ** 2  # [batch_size, seq_len, 2]
        return [seq_output_tail, subject_preds]

    def predict_object(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output_tail, subject_ids = inputs
            subject = self.extract_subject([seq_output_tail, subject_ids])
            output = self.condLayerNorm(seq_output_tail, subject)
            output = (torch.sigmoid(self.linear2(output))) ** 4
            object_preds = output.reshape(*output.shape[:2], len(predicate2id), 2)
        return object_preds

    def calculate_sharpness(self, inputs, targets, epsilon=1e-5):
        # calculate original loss
        outputs = self(*inputs)
        loss = self.compute_loss(outputs, targets)
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        grad_norm = sum(grad.pow(2).sum() for grad in grads)

        # apply small perturbation to parameters of the model
        with torch.no_grad():
            for p, grad in zip(self.parameters(), grads):
                p += epsilon * grad / (grad_norm + 1e-6)

        # calculate the loss after perturbation
        perturbed_outputs = self(*inputs)
        perturbed_loss = self.compute_loss(perturbed_outputs, targets)

        # restore the original model parameters
        with torch.no_grad():
            for p, grad in zip(self.parameters(), grads):
                p -= epsilon * grad / (grad_norm + 1e-6)

        # return the difference between the perturbed loss and the original loss as the sharpness estimate
        return perturbed_loss - loss