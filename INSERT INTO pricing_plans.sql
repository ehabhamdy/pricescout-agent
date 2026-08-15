INSERT INTO pricing_plans 
(company_name, plan_name, input_tokens, output_tokens, currency, billing_period, features, limitations, source_query) 
VALUES 
('Fireworks', 'Base model <4B parameters', 0.1, null, 'USD', 'per million tokens', 'For models with less than 4 billion parameters', null, 'Pricing pages of Fireworks and Groq'), 
('Fireworks', 'Base model 4B-16B parameters', 0.2, null, 'USD', 'per million tokens', 'For models between 4 and 16 billion parameters', null, 'Pricing pages of Fireworks and Groq'), 
('Fireworks', 'Base model >16B parameters', 0.9, null, 'USD', 'per million tokens', 'For models with more than 16 billion parameters', null, 'Pricing pages of Fireworks and Groq'), 
('Fireworks', 'MoE 0B - 56B parameters (e.g. Mixtral 8x7B)', 0.5, null, 'USD', 'per million tokens', 'Mixture of Experts for models between 0 and 56B parameters', null, 'Pricing pages of Fireworks and Groq'), 
('Fireworks', 'MoE 56.1B - 176B parameters (e.g. DBRX, Mixtral 8x22B)', 1.2, null, 'USD', 'per million tokens', 'Mixture of Experts for models between 56.1B and 176B parameters', null, 'Pricing pages of Fireworks and Groq'), 
('Groq', 'AI ModelGPT OSS 20B 128k', 0.075, 0.3, 'USD', 'per million tokens', 'Speed: 1,000 tokens per second', null, 'Pricing pages of Fireworks and Groq'), 
('Groq', 'AI ModelGPT OSS Safeguard 20B', 0.075, 0.3, 'USD', 'per million tokens', 'Speed: 1,000 tokens per second', null, 'Pricing pages of Fireworks and Groq'), 
('Groq', 'AI ModelGPT OSS 120B 128k', 0.15, 0.6, 'USD', 'per million tokens', 'Speed: 500 tokens per second', null, 'Pricing pages of Fireworks and Groq'), 
('Groq', 'AI ModelKimi K2-0905 1T 256k', 1.0, 3.0, 'USD', 'per million tokens', 'Speed: 200 tokens per second', null, 'Pricing pages of Fireworks and Groq')