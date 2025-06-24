Fold	BASELINE	ModelSmall	ModelMedium	ModelLarge	ModelSuper
1	8	18	10	10	5
2	17	13	13	12	5
3	10	15	11	17	11

for predicting the special 9 features

class BASELINE(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim):
        super().__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, out_dim)
    def forward(self, x):
        z = self.embed(x)
        h = self.layers(z)
        return self.head(h)

class ModelSmall(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class ModelMedium(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

class ModelLarge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

# New SUPER size model (approximately 2x larger than Large)
class ModelSuper(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)



    [I 2025-06-19 21:34:14,602] Trial 20 finished with value: 1.3320440957904276 and parameters: {'colsample_bytree': 0.6523896204219245, 'learning_rate': 0.01825851493012502, 'min_child_samples': 19, 'min_child_weight': 0.0734122046534916, 'n_estimators': 1006, 'num_leaves': 194, 'reg_alpha': 21.38050031937534, 'reg_lambda': 55.32830737909552, 'subsample': 0.6891921395606442, 'max_depth': 12}. Best is trial 20 with value: 1.3320440957904276.


    [I 2025-06-19 15:46:52,859] Trial 24 finished with value: 2.7496092870631177 and parameters: {'colsample_bylevel': 0.6270804772286696, 'colsample_bynode': 0.2539493376667614, 'colsample_bytree': 0.8859795017649658, 'gamma': 1.0549203532317613, 'learning_rate': 0.028397302540593607, 'max_depth': 20, 'max_leaves': 16, 'min_child_weight': 20, 'n_estimators': 2484, 'subsample': 0.08914129080676321, 'reg_alpha': 35.173890261682324, 'reg_lambda': 50.20015803720688}. Best is trial 24 with value: 2.7496092870631177.