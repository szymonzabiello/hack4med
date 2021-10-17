"""End-to-end data preparation and modelling."""

import shutil
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Probabilistic Programming
import pymc3 as pm
import theano.tensor as tt
import arviz as az


id_cols = ["LP.", "ID_LAB"]
basic_bio_codes = [
    'N58.11.11342_PCT', 'I81.11.1112_CRP', 'G49.122.1113_DD', 'M05_IL-6', 'O59_TNHS', 
    'N11.126.20.1CITO_MLECZ', 'M37.11.191_KREA', 'C55.103.02_WBC', 'C55.103.02_PLT', 
]
basic_adm_vars = [
    'WIEK', 'PLEC', 'WZROST', 'MASA_CIALA', 'BMI', 'RRS', 'RRD', 'ODDECH', 'AS',  "PO2_ATM", 
    'NT', 'DM', 'ASTMA', 'POCHP', 'HF', 'AF', 'UDAR', 'CHD', 'MI', 'ZAP_PLUC', 'PCHN',
    'DEKSAMETEZON', 'HDCZ', 'BB', 'STATYNA', 'ASA', 'NOAC', 'MRA', 'ACE', 'SARTANY', 'CA_BLOKER',
]
basic_target = "ZGON_LUB_OIT"


def load_training_data(s3: str = "s3://epam-hack4med-dataset") -> pd.DataFrame:
    """Loads and flattens the training data."""
    # Load labels
    df_labels = pd.read_csv(f"{s3}/CRACoV-ETYKIETY.csv")
    df_labels[id_cols] = df_labels[id_cols].astype(int)
    df_labels = df_labels.set_index(id_cols)
    labels = df_labels[[basic_target]]
    idx = labels.index

    # Load hospital admission file (PRZYJECIE)
    df_admission = pd.read_csv(f"{s3}/CRACoV-PRZYJECIE.csv")
    binary_adm_vars = [x for x in basic_adm_vars if df_admission[x].isin(["Tak", "Nie"]).any()]
    other_adm_vars = [x for x in basic_adm_vars if x not in binary_adm_vars]
    adm = df_admission.copy()
    adm = adm[id_cols + binary_adm_vars + other_adm_vars]
    adm = adm.set_index(id_cols).reindex(idx)
    
    # Load biochem analyses
    biochem_raw = pd.read_csv(f"{s3}/CRACoV-BIOCHEMIA.csv", parse_dates=['DATA_WYK']).sort_values('DATA_WYK')
    biochem = (
        biochem_raw.loc[biochem_raw.KOD.isin(basic_bio_codes)]
       .pivot_table(index=['LP.', 'ID_LAB'], columns='KOD', values='WYNIK', aggfunc='first')
       .reindex(idx)
    )
    # Merge it all together
    Xy_raw = pd.concat([labels, adm, biochem], axis='columns')
    return Xy_raw


def cleanup_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe & removes many data issues."""
    df = df.copy()
    
    # Map column names
    if "PO2" in df.columns:
        df["PO2_ATM"] = df["PO2"]

    # Fix fuzzy binary values
    _bm = {"Tak": 1, "Nie": 0, "Nie wiadomo": -1, np.nan: -1}

    def map_binary_values(x) -> int:
        """Map binary values within the column."""
        return _bm.get(x, -1)

    _bin_vars = [x for x in df.columns if df[x].isin(["Tak", "Nie", "Nie wiadomo"]).any()]
    for col in _bin_vars:
        df[col] = df[col].map(map_binary_values)
        
    # Fix numeric columns in biochem 
    # We add "*_out_of_bound" to all columns for consistency with potential test data being OOB
    for col in basic_bio_codes:
        if df[col].dtype == 'object':
            df[f'{col}_out_of_bound'] = df[col].str.match('.*(<|>).*').fillna(False).astype(int)
            df[col] = df[col].str.replace(',', '.').str.replace('[^0-9.]', '', regex=True).astype('float64')
        else:
            warnings.warn(f"Column within bounds: {col!r}")
            df[f'{col}_out_of_bound'] = 0
    
    # Fix height and weight being swapped
    t1 = df["WZROST"] < df["MASA_CIALA"]
    df.loc[t1, ['WZROST', 'MASA_CIALA']] = df.loc[t1, ['MASA_CIALA', 'WZROST']].values
    df.loc[t1, "BMI"] = df.loc[t1, "MASA_CIALA"] / (df.loc[t1, "WZROST"]/100)**2
    
    # Fix out-of-bounds values - hacky but fast
    df.loc[df["MASA_CIALA"] < 10, ["MASA_CIALA", "BMI"]] = np.nan
    df.loc[df["WZROST"] < 50, ["WZROST", "BMI"]] = np.nan
    df.loc[df["ODDECH"] > 60, "ODDECH"] = np.nan
    df.loc[df["PO2_ATM"] < 10, "PO2_ATM"] = np.nan
    df.loc[df["C55.103.02_WBC"] > 60, "C55.103.02_WBC"] = np.nan  # unsure

    return df


class EpamModel(object):
    """Bayesian model by the EPAM team."""
    
    v_known = ['PLEC']
    v_fuzzy = ['NT', 'DM', 'ASTMA', 'POCHP', 'HF', 'AF', 'UDAR', 'CHD', 'MI',
           'ZAP_PLUC', 'PCHN', 'DEKSAMETEZON', 'HDCZ', 'BB', 'STATYNA', 'ASA',
           'NOAC', 'MRA', 'ACE', 'SARTANY', 'CA_BLOKER']
    v_float_adm = ['WIEK', 'WZROST', 'MASA_CIALA', 'BMI', 'RRS', 'RRD', 'ODDECH', 'AS',
           'PO2_ATM']
    v_float_bio = ['C55.103.02_PLT', 'C55.103.02_WBC', 'G49.122.1113_DD',
           'I81.11.1112_CRP', 'M05_IL-6', 'M37.11.191_KREA',
           'N11.126.20.1CITO_MLECZ', 'N58.11.11342_PCT', 'O59_TNHS']
    v_oob_bio = ['N58.11.11342_PCT_out_of_bound', 'I81.11.1112_CRP_out_of_bound',
           'G49.122.1113_DD_out_of_bound', 'M05_IL-6_out_of_bound',
           'O59_TNHS_out_of_bound', 'N11.126.20.1CITO_MLECZ_out_of_bound',
           'M37.11.191_KREA_out_of_bound', 'C55.103.02_WBC_out_of_bound',
           'C55.103.02_PLT_out_of_bound']
    
    def __init__(self, *, oversample: bool = True):
        self.oversample = oversample  # used only during fitting
        # self.C_mean_
        # self.C_std_
        # self.ifd_
        
    def build_model(self, X: pd.DataFrame, y: pd.DataFrame = None) -> pm.Model:
        """Builds the probabilistic model."""
        idx = X.index
        
        if y is None:
            y = pd.Series(0, index=idx)
        elif self.oversample:  # only if y is given
            n_pos = (y == 1).sum()
            n_neg = (y == 0).sum()
            to_add = int(np.ceil(n_neg/n_pos) - 1)
            # print(n_pos, n_neg, to_add)
            if to_add > 4:
                to_add = 4
            for i in range(to_add):
                idx = idx.append(y[y==1].index)
            X = X.loc[idx]
            y = y.loc[idx]
        
        A = X[self.v_known + self.v_oob_bio]
        B_vals = X[self.v_fuzzy]
        B_mask = (B_vals == -1).astype(int)
        C_raw = X[self.v_float_adm + self.v_float_bio]
        # C_scaled = (C_raw - self.C_mean_) / self.C_std_ 
        C_scaled = np.log1p(C_raw/self.C_mean_)
        C_scaled[~np.isfinite(C_scaled)] = np.nan
        C_vals = C_scaled.fillna(0)
        C_mask = C_scaled.isnull().astype(int)
        
        coords = {"idx": idx, "a": A.columns, "b": B_vals.columns, "c": C_vals.columns}
        with pm.Model(coords=coords) as m:
            pm.Data("A", A, dims=["idx", "a"])
            pm.Data("B_vals", B_vals, dims=["idx", "b"])
            pm.Data("B_mask", B_mask, dims=["idx", "b"])
            pm.Data("C_vals", C_vals, dims=["idx", "c"])
            pm.Data("C_mask", C_mask, dims=["idx", "c"])
            pm.Data("y", y, dims=["idx"])

            pm.Normal("avg", mu=0, sd=1)

            pm.Beta("h_a_incl", alpha=1, beta=4)
            pm.Normal("a_coef_raw", mu=0, sd=1, dims=["a"])
            pm.Bernoulli("a_incl", p=m["h_a_incl"], dims=["a"])
            pm.Deterministic("a_coef", m['a_coef_raw'] * m['a_incl'], dims=["a"])
            
            pm.Normal("b_vals_coef", mu=0, sd=1, dims=["b"])
            pm.Normal("b_mask_coef_raw", mu=0, sd=1, dims=["b"])
            pm.Beta("h_b_mask_incl", alpha=1, beta=4)
            pm.Bernoulli("b_mask_incl", p=m["h_b_mask_incl"], dims=["b"])
            pm.Deterministic("b_mask_coef", m['b_mask_coef_raw'] * m['b_mask_incl'], dims=["b"])
            
            pm.Normal("c_vals_coef", mu=0, sd=1, dims=["c"])
            pm.Normal("c_mask_coef_raw", mu=0, sd=1, dims=["c"])
            pm.Beta("h_c_mask_incl", alpha=1, beta=4)
            pm.Bernoulli("c_mask_incl", p=m["h_c_mask_incl"], dims=["c"])
            pm.Deterministic("c_mask_coef", m['c_mask_coef_raw'] * m['c_mask_incl'], dims=["c"])
            unprob = pm.Deterministic(
                "logit",
                m['avg']
                + tt.dot(m["A"], m["a_coef"])
                + tt.dot(m["B_vals"] * (1 - m['B_mask']), m["b_vals_coef"])
                + tt.dot(m["B_mask"], m["b_mask_coef"])
                + tt.dot(m["C_vals"] * (1 - m['C_mask']), m["c_vals_coef"])
                + tt.dot(m["C_mask"], m["c_mask_coef"])
            )
            pm.Bernoulli("y_pred", p = tt.nnet.sigmoid(unprob), dims=['idx'], observed=m['y'])

            m.graph = pm.model_to_graphviz()

        return m
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Fits the model and performs in-sample predict_proba."""
        # I dislike this convention, but sklearn does it, so... sure.
        self.C_mean_ = X[self.v_float_adm + self.v_float_bio].mean(axis='rows')
        self.C_std_ = X[self.v_float_adm + self.v_float_bio].std(axis='rows')
        
        with self.build_model(X, y) as m:
            start_vals = pm.find_MAP()
            _prior = pm.sample_prior_predictive()
            ifd = pm.sample(draws=200, chains=4, start=start_vals, return_inferencedata=True)
            _posterior_pred = pm.fast_sample_posterior_predictive(ifd)
            # _posterior_pred = pm.sample_posterior_predictive(ifd)
            ifd.extend(az.from_pymc3(prior=_prior, posterior_predictive=_posterior_pred))
            ppc_ins = ifd.posterior_predictive['y_pred']
            y_ins = ppc_ins.mean(["chain", "draw"]).to_series()
        
        self.ifd_ = ifd
        if self.oversample:
            y_ins = y_ins.iloc[:len(y)]  # select in-sample values, without duplicates
        return y_ins
        
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Probabilistic prediction."""
        ifd = self.ifd_
        with self.build_model(X) as m:
            # _ppc_oos = pm.sample_posterior_predictive(ifd)
            _ppc_oos = pm.fast_sample_posterior_predictive(ifd)
            ifd_oos = az.from_pymc3(posterior_predictive=_ppc_oos)
            ppc_oos = ifd_oos.posterior_predictive['y_pred']
            y_oos = ppc_oos.mean(["chain", "draw"]).to_series()
        return y_oos
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Point estimate of prediction."""
        # FIXME: Select a threshold? Should we do that during fitting?
        threshold = 0.5
        y_proba = self.predict_proba(X)
        y_point = (y_proba > threshold).astype(int)
        return y_proba
        
    def plot_forest(self):
        """Creates a Forest plot of coefficient values across chains."""
        ax, = az.plot_forest(self.ifd_, var_names=["avg", "a_coef", "b_vals_coef", "b_mask_coef",  "c_vals_coef", "c_mask_coef"])
        ax.axvline(0, linestyle=':', color='black')
        # return ax

    def plot_trace(self):
        """Creates a trace plot of coefficients, to check convergence."""
        az.plot_trace(self.ifd_)

    def save(self, path):
        """Saves the model and other required data.
        
        Note that we can't just use pickle or cloudpickle, as the Theano graph that PyMC3 uses can't really be pickled.
        """
        if isinstance(path, str) and path.startswith("s3://"):
            raise NotImplementedError("TODO: Implement saving to s3")
        path = Path(path).resolve()
        
        # Make a folder to save things into
        if path.suffix == ".zip":
            folder = path.parent / f"tmp_{path.stem}"
        else:
            folder = path
        folder.mkdir(parents=True, exist_ok=False)
        
        # Save the trace - ugly hack needed due to multiindex not being supported yet
        ifd = self.ifd_.copy()
        ifd.constant_data = ifd.constant_data.reset_index('idx')
        ifd.observed_data = ifd.observed_data.reset_index('idx')
        ifd.prior_predictive = ifd.prior_predictive.reset_index('idx')
        ifd.posterior_predictive = ifd.posterior_predictive.reset_index('idx')
        ifd.log_likelihood = ifd.log_likelihood.reset_index('idx')
        az.to_netcdf(ifd, folder / "ifd.nc")
        
        # Save other data
        self.C_mean_.rename("C_mean_").to_csv(folder / "C_mean_.csv")
        self.C_std_.rename("C_std_").to_csv(folder / "C_std_.csv")
        
        if path.suffix == ".zip":
            # Save to zip
            zf = ZipFile(path, mode='w')
            for file in folder.glob("*"):
                zf.write(folder / file.name, arcname=file.name)
            zf.close()
            shutil.rmtree(folder)
        
    def load(self, path):
        """Loads the model from a path. You should set the same parameters in __init__."""
        if isinstance(path, str) and path.startswith("s3://"):
            raise NotImplementedError("TODO: Implement loading from s3")
        
        # Make a folder to save things into
        path = Path(path).resolve()
        
        if path.suffix == ".zip":
            folder = path.parent / f"tmp_{path.stem}"
            folder.mkdir(parents=True, exist_ok=False)
            
            zf = ZipFile(path, mode='r')
            zf.extractall(folder)
        else:
            folder = path
        
        # Load the trace - ugly hack needed due to multiindex not being supported yet
        ifd = az.from_netcdf(folder / "ifd.nc")
        ifd.constant_data = ifd.constant_data.set_index({"idx": id_cols})
        ifd.observed_data = ifd.observed_data.set_index({"idx": id_cols})
        ifd.prior_predictive = ifd.prior_predictive.set_index({"idx": id_cols})
        ifd.posterior_predictive = ifd.posterior_predictive.set_index({"idx": id_cols})
        ifd.log_likelihood = ifd.log_likelihood.set_index({"idx": id_cols})
        self.ifd_ = ifd
        
        # Load other data
        self.C_mean_ = pd.read_csv(folder / "C_mean_.csv", index_col=0)["C_mean_"]
        self.C_std_ = pd.read_csv(folder / "C_std_.csv", index_col=0)["C_std_"]
        
        # Ideally we would save parameters as YAML or something, but no time
        if path.suffix == ".zip":
            shutil.rmtree(folder)

def main():
    """Creates the prediction for evaluation."""
    
    # Load the model
    model = EpamModel()
    model.load("bayes_1.zip")
    
    # Load and clean/prepare test data 
    x_test = pd.read_csv('BAZA_VALID_INPUT.csv')
    x_test_clean = cleanup_df(x_test)
    
    # Predict
    # FIXME: This currently does probabilistic prediction only!
    y_pred = model.predict(x_test_clean)
    
    with open('output.txt', 'w+') as f:
        for label in y_pred:
            f.write(f'{label}\n')

            
try:
    import ipywidgets as w
    
    def make_interactive(model: EpamModel, Xy_raw: pd.DataFrame):
        """Makes an interactive example, with initial values taken from a real patient."""
        
        def inner(base_example: int = 0):
            """Inner function that does the work."""
            EX = Xy_raw.iloc[base_example]
            print("Base Example IDs:")
            print((Xy_raw.index[[base_example]]).to_frame().reset_index(drop=True).iloc[0].rename(index=base_example))

            def get_widget(col: str):
                V = EX[col]
                if 'Tak' in Xy_raw[col].values.tolist():
                    zz = ["Tak", "Nie", "Nie wiadomo"]
                    return w.Select(options=zz, value="Nie wiadomo" if (V not in zz) else V)

                d = Xy_raw[col].dtype
                if d == "int64":
                    return w.IntSlider(min=Xy_raw[col].min(), max=Xy_raw[col].max(), value=V)
                elif d == "float64":
                    return w.FloatText(value=V)#Xy_raw[col].mean())

                def _f(s):
                    try:
                        return float(s)
                    except:
                        return np.nan
                avg_val = Xy_raw[col].apply(_f).mean()
                if pd.isnull(V):
                    V = str(avg_val)
                return w.Text(value=V)


            def make_pred(**vals):
                my_X_raw = pd.Series(vals).to_frame().T
                my_X_raw = my_X_raw.astype(Xy_raw.drop(columns=[basic_target]).dtypes)
                my_X = cleanup_df(my_X_raw)#.astype(X.dtypes)

                ifd = model.ifd_
                with model.build_model(my_X) as m:
                    _ppc_oos = pm.fast_sample_posterior_predictive(ifd)
                    ifd_oos = az.from_pymc3(posterior_predictive=_ppc_oos)
                    ppc_oos = ifd_oos.posterior_predictive['y_pred']
                    y_oos = ppc_oos.mean(["chain", "draw"]).to_series()

                # y_prob = model.predict_proba(my_X)
                print("Prediction:", y_oos.iloc[0])

            widgets = {k: get_widget(k) for k in Xy_raw.drop(columns=[basic_target]).columns}
            
            w.interact_manual(make_pred, **widgets)
        return w.interact(
            inner, 
            base_example=w.IntText(min=0, max=len(Xy_raw)-1, value=0, description="Base Case")
        )
    
except Exception:  # ImportError or whatever
    warnings.warn("ipywidgets not installed, or other error with interactivity")
    
if __name__ == '__main__':
    main()
